from importlib.metadata import distribution
import torch

from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)
import matplotlib.pyplot as plt

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
from monai.losses import DiceLoss


from models import SegmentationNet, SelectionNet
device_0 = "cuda:0"
device_1 = "cuda:1"



train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

data_dir = "/raid/candi/xiangcen/ahnet/Task07_Pancreas/dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")

D_test = datalist[:21]
D_meta_train = datalist[21:151]
D_meta_select = datalist[151:281]
print(len(D_test), len(D_meta_train), len(D_meta_select))


seg_ds = Dataset(
    data=D_meta_train,
    transform=train_transforms,
)
seg_loader = DataLoader(
    seg_ds, batch_size=4, shuffle=False,
)


selection_ds = Dataset(
    data=D_meta_train,
    transform=train_transforms,
)
selection_loader = DataLoader(
    selection_ds, batch_size=4, shuffle=False,
)





dice_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    reduction="none",
)

def Weighted_MMD(distribution_0, alpha_0, distribution_1, alpha_1):
    return distribution_0.sum()+alpha_0.sum() + distribution_1.sum() + alpha_1.sum()




def result_transfer(l, alpha):
    """
    create two instance of loss (from the f_seg) and alpha (from the f_select)
    on two gpus (f_seg for device_0 and f_select for device_1)

    Args:
        l (torch.tensor): The dice loss created by f_seg on device_0
        alpha (torch.tensor): The representativeness index created by f_select on device_1
    """
    
    l_0 = l.to(device_0)
    l_1 = l.to(device_1)
    alpha_0 = alpha.to(device_0)
    alpha_1 = alpha.to(device_1)
    return l_0, alpha_0, l_1, alpha_1

def gpu_transfer(a, b):
    
    
    a_0 = a.to(device_0)
    a_1 = a.to(device_1)
    b_0 = b.to(device_0)
    b_1 = b.to(device_1)
    return a_0, a_1, b_0, b_1


f_seg = SegmentationNet().to(device_0)
f_select = SelectionNet().to(device_1)
f_seg_optimizer = torch.optim.SGD(f_seg.parameters(), lr=0.1)
f_select_optimizer = torch.optim.SGD(f_select.parameters(), lr=0.1)




def init_data(batch_0, batch_1):

    # sample the data and copy them on two gpus
    img_seg, label_seg = batch_0["image"], batch_0["label"]
    img_select, label_select = batch_1["image"], batch_1["label"]
    # copy and send
    img_seg_0, label_seg_0, img_seg_1 = img_seg.to(device_0), label_seg.to(device_0), img_seg.to(device_1)
    img_select_1, img_select_0, label_select_0 = img_select.to(device_1), img_select.to(device_0), label_select.to(device_0)
    return img_seg_0, label_seg_0, img_seg_1, img_select_1, img_select_0, label_select_0


def train_f_seg(img_seg_0, label_seg_0, img_seg_1, f_seg, f_select, f_seg_optimizer):
    with torch.no_grad():
        f_select.eval()
        alpha_0 = f_select(img_seg_1).to(device_0)
    f_select.train()
    pred_0 = f_seg(img_seg_0)
    loss_0 = torch.mean(dice_loss(pred_0, label_seg_0).flatten(2), dim=1)
    print(loss_0.shape)
    loss_0 = torch.mean(loss_0*alpha_0, dim=0)
    print(loss_0.shape)
    
    loss_0.backward()
    f_seg_optimizer.step()
    return loss_0.item()


def train_f_select(img_select_1, img_select_0, label_select_0, f_select, f_seg, f_select_optimizer):
    with torch.no_grad():
        f_seg.eval()
        pred_0 = f_seg(img_select_0)
        loss_1 = dice_loss(pred_0, label_select_0).flatten(2).to(device_1)
        
    f_seg.train()
    alpha_1 = f_select(img_select_1)
    # find the largest alpha(s)
    top_k_indices = torch.topk(alpha_1, k=1, dim=0)[1].squeeze(0)
    distribution_val, alpha_val =  loss_1[top_k_indices], alpha_1[top_k_indices]
    print(distribution_val.shape, alpha_val.shape, loss_1.shape, alpha_1.shape)
    loss_1 = Weighted_MMD(distribution_val, alpha_val, loss_1, alpha_1)
    loss_1.backward()
    f_select_optimizer.step()
    return loss_1.item()






for batch_0, batch_1 in zip(seg_loader, selection_loader):
    img_seg_0, label_seg_0, img_seg_1, img_select_1, img_select_0, label_select_0 = init_data(batch_0, batch_1)
    # loss = train_f_seg(img_seg_0, label_seg_0, img_seg_1, f_seg, f_select, f_seg_optimizer)
    print(f_select.qkv.weight)
    train_f_select(img_select_1, img_select_0, label_select_0, f_select, f_seg, f_select_optimizer)
    print(f_select.qkv.weight)
    break




#### To Do List
"""
Define a helper function for the dice loss so that it could produce dice loss for all the individual loss inside a batch
"""
