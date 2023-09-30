import torch.nn as nn
import torch
from typing import List


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2, alpha: float = 0.25, num_classes: int = 4):
        """
        Based on https://arxiv.org/pdf/1708.02002.pdf

        :param gamma: gamma hyperparam of RetinaNet paper
        :param alpha: apha hyperparam of RetinaNet paper
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Pred tensor is of shape width x height x num_achors * num_classes. We need to extract the prediction such that
        the tensor is of shape width x height x num_anchors where each anchor predicts a class

        :param pred: Class prediction output of the model
        :param gt: Ground truth label
        :return: Focal loss for class prediction
        """

        pred = pred.permute(0,2,3,1)
        obj_mask = (gt != 0).float()
        noobj_mask = (gt == 0).float()
        num_obj = obj_mask.sum()

        pred = torch.sigmoid(pred)
        pos_loss = - self.alpha * torch.pow(1 - pred, self.gamma) * torch.log(pred + 1e-7) * obj_mask
        neg_loss = - (1 - self.alpha) * torch.pow(pred, self.gamma) * torch.log(1 - pred + 1e-7) * noobj_mask
        loss = pos_loss.sum() + neg_loss.sum()

        if num_obj != 0:
            loss /= num_obj

        return loss


class BoxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, anchors: torch.Tensor):

        pred = pred.permute(0, 2, 3, 1)
        pred = pred.view(*pred.shape[:-1],  len(anchors), 4)
        gt = gt.view(*gt.shape[:-1], len(anchors), 4)
        obj_mask = (gt != 0).float()

        pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])
        gt[..., 2:4] = torch.log(gt[..., 2:4] + 1e-7) / anchors

        loss = self.mse(pred, gt)
        loss *= obj_mask
        loss = loss.mean()
        return loss


class RetinaLoss(nn.Module):
    def __init__(self, num_classes: int = 4, gamma: int = 2, alpha: float = 0.25, device: str ="cuda",
                 w1: float = 1, w2: float = 10):
        super().__init__()
        self.focal_loss = FocalLoss(gamma, alpha, num_classes)
        self.box_loss = BoxLoss()
        self.device = device
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred: List[torch.Tensor], gt: List[torch.Tensor], anchors: torch.Tensor):
        """
        Computes Loss for RetinaNet

        :param pred: List of class and box prediction
        :param gt: List of class and box target
        :return: Sum of focal and boxloss
        """

        fc_loss = self.focal_loss(pred[0].to(self.device), gt[:, 0].to(self.device))
        box_loss = self.box_loss(pred[1].to(self.device), gt[:, 1].to(self.device), anchors)

        return self.w1 * fc_loss + self.w2 * box_loss, fc_loss, box_loss




