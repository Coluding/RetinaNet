from torch.utils.data import Dataset
import torch
import albumentations as A
from typing import List
import glob
import numpy as np
from PIL import Image

from utils import iou_width_height, test_transforms


class RetinaDataset(Dataset):
    def __init__(self,
                 label_directory: str,
                 img_directory: str,
                 anchors: List[List[float]],
                 image_size: int = 416,
                 S: List[int] = [32, 16, 8],
                 C: int = 4,
                 transforms: A.Compose = None,
                 ):

        self.label_dir = label_directory
        self.img_directory = img_directory
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.image_size = image_size
        self.S = S
        self.C = C
        self.transforms = transforms
        self.ignore_iou_thresh = 0.5

        self.labels = list(sorted(glob.glob(label_directory + "/*.txt")))
        self.imgs = list(sorted(glob.glob(img_directory + "/*.jpg")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        label_path = self.labels[index]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1)
        img_path = self.imgs[index]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms:
            augs = self.transforms(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        # We will predict 2 tensors of shape batch_size x size//stride x size//stride x num_anchors * classes and
        # batch_size x size//stride x size//stride x num_anchors * 4 fr the bounding box at each scale
        # each class is treated as a separate bianry prediction problem --> no background class and no softmax!!
        targets_class = [torch.zeros(s, s, self.num_anchors_per_scale * self.C)
                         for s in self.S]

        targets_bbox = [torch.zeros(s, s, 4 * self.num_anchors_per_scale)
                        for s in self.S]

        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, w, h, class_label = box

            has_anchor = [False] * self.num_anchors_per_scale

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_scale_idx = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                scale_img_size = S
                i, j = int(x * scale_img_size), int(y * scale_img_size)
                anchor_taken = targets_class[scale_idx][i, j, anchor_scale_idx]

                if not has_anchor[scale_idx] and not anchor_taken:
                    cell_x, cell_y = (x * scale_img_size) - i, (y * scale_img_size) - j
                    width_cell = scale_img_size * w
                    height_cell = scale_img_size * h

                    targets_bbox[scale_idx][i, j, (anchor_scale_idx * 4):(anchor_scale_idx * 4) + 4] = torch.tensor(
                        [cell_x, cell_y, width_cell, height_cell]
                    )
                    targets_class[scale_idx][i, j, (anchor_scale_idx * 4) + int(class_label)] = 1
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets_class[scale_idx][i, j, anchor_scale_idx] = -1 # ignore prediction

        tensors_per_scale = [torch.stack((torch.tensor(targets_class[i]), torch.tensor(targets_bbox[i])))
                             for i in range(len(targets_bbox))]
        return image, tuple(tensors_per_scale)
