# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Note that `downscale_label_ratio` method is adapted from: https://github.com/lhoyer/DAFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import functools

# import mmcv
import numpy as np
import torch.nn.functional as F

import numpy as np
import json
import pickle
import yaml

def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        elif class_weight.endswith('.pkl'):
            with open(class_weight, 'rb') as f:
                class_weight = pickle.load(f)
        elif class_weight.endswith('.json'):
            with open(class_weight, 'r') as f:
                class_weight = json.load(f)
        elif class_weight.endswith('.yaml') or class_weight.endswith('.yml'):
            with open(class_weight, 'r') as f:
                class_weight = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {class_weight}")
    
    return class_weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor >= 1
    if scale_factor == 1:
        return gt.clone()
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # o/w next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index):
    # 如果掩码的分辨率与特征图不一致，进行下采样
    if mask.shape[-2:] != feat.shape[-2:]:
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest').long()

    # 如果掩码有多个通道 (例如 one-hot 编码)，需要先合并这些通道
    if mask.dim() == 4 and mask.size(1) > 1:
        mask = mask.argmax(dim=1)  # 通过 argmax 转换为单通道掩码

    # 确保 feat 的形状为 [batch_size * height * width, channels]
    batch_size, channels, height, width = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, channels)  # (batch_size * height * width, channels)
    
    # 展开 mask 为 [batch_size * height * width]
    mask = mask.view(-1)

    # 检查 mask 和 feat 的第一个维度是否匹配
    assert feat.size(0) == mask.size(0), f"Feature shape {feat.size()} and mask shape {mask.size()} do not match!"
    
    # 过滤掉忽略的索引
    msk = (mask != ignore_index)
    mask = mask[msk]
    feat = feat[msk]  # 使用掩码过滤特征图
    return feat, mask



def proto_reg(feat,
              mean=None,
              contrast_temp=100.,
              contrast_norm=None,
              **kwargs):
    assert mean is not None, 'Parameter `mean` required'
    assert contrast_norm is not None, 'Parameter `contrast_norm` required'
    assert not mean.requires_grad
    assert feat.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    mean_feat = torch.mean(feat, 0, keepdim=True)

    # feat (1, A) x Ave (A, C)
    proto_sim = mean_feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp

    loss = torch.sum(torch.softmax(proto_sim, dim=1).log()) / contrast_norm

    return loss


def proto_contrastive(feat,
                      mask,
                      mean=None,
                      index=-1,
                      contrast_temp=100.,
                      use_avg_pool=True,
                      scale_min_ratio=0.75,
                      num_classes=19,
                      weight=None,
                      class_weight=None,
                      reduction='mean',
                      avg_factor=None,
                      reg_weight=0,
                      ignore_index=255,
                      **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
    
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert mean is not None, 'Parameter `mean` required'
    # assert not mean.requires_grad
    assert isinstance(mean, torch.Tensor), 'mean should be a Tensor'
    assert not mean.requires_grad, "mean should not require gradient"
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    # feat (N, A) x Ave (A, C)
    proto_sim = feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp

    # The wrapper function for :func:`F.cross_entropy`
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        proto_sim,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


def dist_contrastive(feat,
                     mask,
                     mean=None,
                     covariance=None,
                     ratio=1.0,
                     index=-1,
                     contrast_temp=100.,
                     use_avg_pool=True,
                     scale_min_ratio=0.75,
                     num_classes=19,
                     weight=None,
                     class_weight=None,
                     reduction='mean',
                     avg_factor=None,
                     reg_weight=0,
                     ignore_index=255,
                     **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        assert isinstance(covariance, (list, dict)), f'covariance list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
        covariance = covariance[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert mean is not None, 'Parameter `mean` required'
    assert covariance is not None, 'Parameter `covariance` required'
    assert not mean.requires_grad
    assert not covariance.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    # feat (N, A) x Ave (A, C)
    temp1 = feat.mm(mean.permute(1, 0).contiguous())
    # feat (N, A)^2 x CoVariance (A, C)
    covariance = covariance * ratio / contrast_temp
    temp2 = 0.5 * feat.pow(2).mm(covariance.permute(1, 0).contiguous())

    logits = temp1 + temp2
    logits = logits / contrast_temp

    # The wrapper function for :func:`F.cross_entropy`
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    ce_loss = F.cross_entropy(
        logits,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # calculate the second term using temp2 cache
    # 1. select the corresponding CoVariance of the positive class
    # q_i^T \Sigma q_i / \tau -> q_i^2 mul \Sigma
    jcl_loss = 0.5 * torch.sum(feat.pow(2).mul(covariance[mask]), dim=1) / contrast_temp

    loss = ce_loss + jcl_loss

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


def bank_contrastive(feat,
                     mask,
                     bank=None,
                     mean=None,
                     index=-1,
                     contrast_temp=100.,
                     use_avg_pool=True,
                     scale_min_ratio=0.75,
                     num_classes=19,
                     weight=None,
                     reduction='mean',
                     avg_factor=None,
                     reg_weight=0,
                     ignore_index=255,
                     **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(bank, (list, dict)) \
               and isinstance(bank[index], deque), f'bank list expected for index={index}'
        feat = feat[index]
        bank = bank[index]
        if reg_weight > 0.:
            assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
            mean = mean[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert bank is not None, 'Parameter `bank` required'
    if reg_weight > 0.:
        assert mean is not None, 'Parameter `mean` required'
        assert not mean.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    loss = []
    # calculate per class
    for cls in range(num_classes):
        cls_filter = (mask == cls)
        cls_feat = feat[cls_filter]  # NcxA
        pos, neg = [], []
        for idx in range(num_classes):
            idx_bank = list(bank[idx])
            cls_bank = torch.cat(idx_bank, dim=0)
            bank_sim = cls_feat.mm(cls_bank.permute(1, 0).contiguous()) / contrast_temp
            if idx == cls:
                pos = bank_sim  # NcxMp
            else:
                neg.append(bank_sim.mean(1, keepdim=True))  # NcxMn -> Ncx1
        neg = torch.cat(neg, dim=1)  # Ncx(C-1)
        exp_pos = pos.exp()  # NcxMp
        sum_exp_neg = neg.exp().sum(1, keepdim=True)  # Ncx1
        softmax_term = exp_pos / (exp_pos + sum_exp_neg)  # NcxMp
        cls_loss = - softmax_term.log().mean(dim=1)  # Nc
        loss.append(cls_loss)

    loss = torch.cat(loss, dim=0)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


# @LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    """ContrastiveLoss.

    Args:
        use_dist (bool, optional): Whether to use distribution based contrastive loss.
            Defaults to False.
        use_bank (bool, optional): Whether to use memory bank based contrastive loss.
            Defaults to False.
        use_reg (bool, optional): Whether to use regularization term.
            Defaults to False.
        use_avg_pool (bool, optional): Whether to use average pooling for down sampling.
            Defaults to True.
        contrast_temp (double, optional): Temperature used in contrastive loss.
            Defaults to 100.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_dist=False,
                 use_bank=False,
                 use_reg=False,
                 use_avg_pool=True,
                 scale_min_ratio=0.75,
                 num_classes=None,
                 contrast_temp=100.,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 reg_relative_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        assert (use_dist is False) or (use_bank is False)
        assert num_classes is not None
        self.use_dist = use_dist
        self.use_bank = use_bank
        self.use_reg = use_reg
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.contrast_temp = contrast_temp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_weight = reg_relative_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_dist:
            self.contrast_criterion = dist_contrastive
        elif self.use_bank:
            self.contrast_criterion = bank_contrastive
        else:
            self.contrast_criterion = proto_contrastive

    def forward(self,
                feat,
                mask,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        # Parameters mean, covariance are sometimes required
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = feat.new_tensor(self.class_weight)
        else:
            class_weight = None
        if isinstance(feat, list):
            if not isinstance(self.loss_weight, list):
                self.loss_weight = [self.loss_weight for _ in range(len(feat))]
            loss_contrast = [self.loss_weight[i] * self.contrast_criterion(
                feat,
                mask,
                weight=weight,
                index=i,
                contrast_temp=self.contrast_temp,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                reg_weight=self.reg_weight if self.use_reg else 0,
                **kwargs) for i in range(len(feat))]
            loss_contrast = sum(loss_contrast)
        else:
            loss_contrast = self.loss_weight * self.contrast_criterion(
                feat,
                mask,
                weight=weight,
                contrast_temp=self.contrast_temp,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                reg_weight=self.reg_weight if self.use_reg else 0,
                **kwargs)
        return loss_contrast