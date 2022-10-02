import torch

from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

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
from loss import create_label, batch_wise_loss, dice_metric
from monai.networks.nets.swin_unetr import SwinUNETR
from model import SelectionNet
from tools import delete_from_gpu, change_zero
#################################################
CUDA_LAUNCH_BLOCKING="1"
# Hyperparameters
device_0 = "cuda:2" # device f_seg lives
device_1 = "cuda:2" # device f_select lives
sigma=5

num_batch = 12
num_of_val = num_batch // 4

data_dir = "./data/Task07_Pancreas/dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")


num_of_test = 21
num_of_segmentation = (281 - 21) // 2
# Partition data
D_test = datalist[:num_of_test]
D_meta_train = datalist[num_of_test : num_of_test + num_of_segmentation]
D_meta_select = datalist[num_of_test + num_of_segmentation : ]
print("Test data: ", len(D_test), ", Segmentation Data: ", len(D_meta_train), ", Selection Data: ", len(D_meta_select))
####################################################
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


select_transforms = Compose(
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
    ]
)




# seg_ds = Dataset(
#     data=D_meta_train,
#     transform=train_transforms,
# )
# seg_loader = DataLoader(
#     seg_ds, batch_size=num_batch, shuffle=True, drop_last=False
# )
selection_ds = Dataset(
    data=D_meta_select,
    transform=select_transforms,
)
selection_loader = DataLoader(
    selection_ds, batch_size=num_batch, shuffle=True, drop_last=True
)

# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to(device_0)
f_seg.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/model/f_seg_v0.pt"))
f_select = SelectionNet().to(device_1)
# f_seg_optimizer = torch.optim.Adam(f_seg.parameters(), lr=0.005)
f_select_optimizer = torch.optim.Adam(f_select.parameters(), lr=0.001)


def init_data(batch_0, batch_1):
    """
    A batch is from one loop of the dataloader iterator
    We need to calculate result of meta segmentation and meta selection dataset 
    by two networks which lives on two gpus, so each gpu get a copy of their corrsponding 
    data that need to be calculated.

    variables with tag 0 or 1 indicate that this variable is on or should 
    locate at device_0 or device_1. The f_seg and f_select are trained at 
    the same time with two gpus. (f_seg on device_0, f_select on device_1)

    Args:
        batch_0 : A batch from meta segmentation dataset (currently on cpu)
        batch_1 : A batch from meta selection dataset (currently on cpu)

    Returns:
        tuple: 6 data
    """
    # sample the data and copy them on two gpus
    img_seg, label_seg = batch_0["image"], batch_0["label"]
    img_select, label_select = batch_1["image"], batch_1["label"]
    
    # For training f_seg network, a copy of the img should send to device_1 to and calculate the alphas by f_select(img)
    # and of course img and label on device_1 to calculate loss by dice(f_seg(img), lable)
    img_seg_0, label_seg_0, img_seg_1 = img_seg.to(device_0), label_seg.to(device_0), img_seg.to(device_1)
    # For training f_select network both img and label should send to f_seg to calculate the 
    # dice metric by dice_metric(f_seg(img), lable), and only the img on device_1 to calculate the alpha
    img_select_1, img_select_0, label_select_0 = img_select.to(device_1), img_select.to(device_0), label_select.to(device_0)
    # Train f_seg : segmentation img (device_0)
    #               segmentation label (device_0)
    #               selection img (device_1)
    # Train f_selection : selection img (device_1)
    #                     selection img (device_0)
    #                     selection label (device_0)
    return img_seg_0, label_seg_0, img_seg_1, img_select_1, img_select_0, label_select_0


def train_f_seg(img_seg_0, label_seg_0, img_seg_1, f_seg, f_select, f_seg_optimizer):
    """
    In general f_seg network's parameter lives on device_0 calculate the alpha values 
    from device_1 and transfer it to device_0 and named those variables alpha_0

    Args:
        img_seg_0 (torch.tensor): segmentation img (device_0)
        label_seg_0 (torch.tensor): segmentation label (device_0)
        img_seg_1 (torch.tensor): selection img (device_1)
        f_seg (nn.Module): segmentaiton network (device_0)
        f_select (nn.Module): selection network (device_1)
        f_seg_optimizer (torch.optim): optimizer for f_seg network

    Returns:
        loss of current batch (step): float
    """
    with torch.no_grad():
        f_select.eval()
        # make sure alpha_0's shape is (B, 1) so that it could multiply loss_0
        # alpha_0 -> [B, 1]
        alpha_0 = f_select(img_seg_1).to(device_0)
        delete_from_gpu(img_seg_1)
    f_seg.train()
    pred_0 = f_seg(img_seg_0)
    # make sure the loss_0' shape is (B, 1) so that it could multiply (1 - alpha)
    loss_0 = batch_wise_loss(pred_0, label_seg_0)
    # loss multiply the (1 - alpha)
    loss_0 = torch.mean(loss_0*(1 - alpha_0), dim=0) # After multiplty (1 - alpha) calculate batch-wise mean 
                                                     # of the loss to make this a number for backprop
    
    # Train the parameter for a step!!!
    loss_0.backward()
    f_seg_optimizer.step()
    f_seg_optimizer.zero_grad()
    # return the loss for monitoring purpose
    return loss_0.item()

def my_celoss(alpha, label):
    loss = -label*torch.log(alpha.softmax(0))
    return loss.mean()

def train_f_select(img_select_1, img_select_0, label_select_0, f_select, f_seg, f_select_optimizer):
    """
    In general f_selection network's parameter lives on device_1 calculate the dice score values 
    from device_1 and transfer it to device_0 and named those variables loss_1

    Args:
        img_select_1 (torch.tensor): selection img (device_1)
        img_select_0 (torch.tensor): selection img (device_0)
        label_select_0 (torch.tensor): selection label (device_0)
        f_select (nn.Module): selection network (device_1)
        f_seg (nn.Module): selection network (device_1)
        f_select_optimizer (torch.optim): f_select network

    Returns:
        float: loss score for current batch
    """
    # calculate the dice score of img_select on device_0 and send it to device_1 as loss_1
    with torch.no_grad():
        f_seg.eval()
        pred_0 = f_seg(img_select_0)
        # loss_1 -> [B, num_class]
        loss_1 = dice_metric(pred_0, label_select_0).to(device_1)
        loss_1 = change_zero(loss_1, 0.001)
        print(loss_1)
        delete_from_gpu(img_select_0, label_select_0)
        label = create_label(num_batch, loss_1)


    f_select.train()
    alpha_1 = f_select(img_select_1)
    print(alpha_1)

    loss_1 = my_celoss(alpha_1, label)
    print(loss_1)

    loss_1.backward()
    f_select_optimizer.step()
    f_select_optimizer.zero_grad()
    return loss_1.item()


loss_function = torch.nn.BCELoss()
if __name__ == "__main__":
    Epochs = 300
    for epoch in range(Epochs):
        for step, batch in enumerate(selection_loader):
            img, label = batch["image"].to(device_1), batch["label"].to(device_1) 
            with torch.no_grad():
                output = f_seg(img)
                # loss_1 -> [B, num_class]
                dice_pred_0 = dice_metric(output, label).to(device_1)
                dice_pred_0 = change_zero(dice_pred_0, 0.001)
                print(dice_pred_0)
                alpha_label = create_label(num_batch, dice_pred_0)
                print(alpha_label)


            img = torch.cat([img, label], dim=1)
            alpha = f_select(img)
            print(alpha)
            loss = loss_function(alpha, alpha_label)
            loss.backward()

            f_select_optimizer.step()
            f_select_optimizer.zero_grad()

            print("This is epoch {}, iteration {}, loss {}".format(epoch, step, loss.item()))
        torch.save(f_select.state_dict(), "/raid/candi/xiangcen/meta_data_select/model/selection_net.pt")
            
