import torch
from einops import repeat

from monai.losses import DiceLoss
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


dice_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    reduction="none",
)

def batch_wise_loss(pred, label):
    # returns the dice loss for each batch
    # [Batch, dice_loss]
    return dice_loss(pred, label).mean(1).flatten(1)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

def dice_metric(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=3, D, H, W]
    '''
    val_labels_list = decollate_batch(y_true)
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_labels_list
    ] # 
    val_outputs_list = decollate_batch(y_pred)
    val_output_convert = [
        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
    ] 
    y_true, y_pred = torch.stack(val_labels_convert), torch.stack(val_output_convert)



    numerator = torch.sum(y_true*y_pred, dim=(2,3,4)) * 2
    denominator = torch.sum(y_true, dim=(2,3,4)) + torch.sum(y_pred, dim=(2,3,4)) + eps


    return numerator / denominator

# def classed_batch_wise_metric(pred, label):
#     # returns the dice loss for each batch and each class
#     # [Batch, class, dice_loss]
#     return dice_metric(pred, label)

def kernel(x, sigma):
    """

    :param x: x is already calculated batch-wised
    :param sigma:
    :return:
    """
    return torch.exp(-x**2 / (2*sigma**2))


def term(l_small, alpha_small, l_big, alpha_big, sigma):
    b_small, _ = l_small.shape
    b_big, _ = l_big.shape

    x_0 = repeat(l_small, "h w -> h w c", c=b_big).permute(0, 2, 1)
    x_1 = repeat(l_big, "h w -> c h w", c=b_small)

    # ||x_0 - x_1|| is of shape (b, b, num_dice)
    l2_norm = torch.norm(x_0 - x_1, dim=-1)
    aa = alpha_small @ alpha_big.t()
    k = kernel(l2_norm, sigma)

    a_small = alpha_small.sum()
    a_big = alpha_big.sum()

    return ((1/a_small) * (1/a_big) * aa * k).sum()


def weighted_mmd(distribution_0, weights_0, distribution_1, weights_1, sigma):
    """
    distribution is of shape (num_sample, num_dice)
    weights is of shape (num_sample 1)
    """
    term_1 = term(distribution_0, weights_0, distribution_0, weights_0, sigma)
    term_2 = term(distribution_1, weights_1, distribution_1, weights_1, sigma)
    term_3 = term(distribution_0, weights_0, distribution_1, weights_1, sigma)
    return term_1 + term_2 - 2*term_3



if __name__ == "__main__":
    l_all = torch.tensor([
        [ 0.45, 0.45],
        [0.9, 0.9],
        [0.1, 0.1],
        [0.6, 0.5],
        [0.48, 0.47],
        [0.943, 0.9432],
        [0.1432, 0.1432],
        [0.6432, 0.5432]
    ])
    alpha_all_bad = torch.tensor([
        [0.1],
        [0.998],
        [0.996],
        [0.994],
        [0.1],
        [0.993],
        [0.999],
        [0.991]
    ])
    alpha_all_good = torch.tensor([
        [0.999],
        [0.001],
        [0.0012],
        [0.0032],
        [0.996],
        [0.004],
        [0.003],
        [0.08]
    ])


    # # a parameter to test the gradient
    p = torch.tensor([1.], requires_grad=True)
    alpha_all = p*alpha_all_good
    l_val = l_all[torch.tensor([0, 4])]
    alpha_val = alpha_all_good[torch.tensor([0, 4])]
    
    
    o = weighted_mmd(l_val, alpha_val, l_all, alpha_all, 0.1)
    o.backward()
    print("good prediction------------------------------")
    print("gradient: ", p.grad, "loss: ", o)
    


##################################################
    p = torch.tensor([1.], requires_grad=True)
    alpha_all = p*alpha_all_bad
    l_val = l_all[torch.tensor([0, 4])]
    alpha_val = alpha_all_bad[torch.tensor([0, 4])]
    
    

    
    o = weighted_mmd(l_val, alpha_val, l_all, alpha_all, 0.1)
    o.backward()
    print("good prediction------------------------------")
    print("gradient: ", p.grad, "loss: ", o)
    
    
