import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import cv2 as cv
import torch
from collections import Counter
from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

scale = 1.1
img_size = config["IMG_SIZE"]

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(img_size * scale)),
        A.PadIfNeeded(
            min_height=int(img_size * scale),
            min_width=int(img_size * scale),
            border_mode=cv.BORDER_CONSTANT
        ),
        A.RandomCrop(width=img_size, height=img_size),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.4),
        A.OneOf([
            A.ShiftScaleRotate(
                rotate_limit=20, p=0.5, border_mode=cv.BORDER_CONSTANT
            ),
            A.Affine(shear=15, p=0.5)
        ]),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.PixelDropout(dropout_prob=0.005),
        A.Normalize(),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo")
)

test_transforms = A.Compose([
    A.LongestMaxSize(max_size=img_size),
    A.PadIfNeeded(
        min_width=img_size, min_height=img_size, border_mode=cv.BORDER_CONSTANT
    ),
    A.Normalize(),
    ToTensorV2(),
],
    bbox_params=A.BboxParams(format="yolo")
)


def iou_width_height(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str = "midpoint"):
    """

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes: List[List[float]], iou_threshold: float,
                        threshold: float, box_format: str = "corners"):
    """

    Applies Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    counter = 0

    while bboxes:
        counter += 1
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes: List[List[float]], true_boxes: List[List[float]], iou_threshold: float = 0.5,
        box_format: str="midpoint", num_classes: int = 5
):
    """

    This function calculates mean average precision (mAP) @AlladinPearson

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image: PIL.Image.Image, boxes: List[List[float]], class_labels: List[str]):
    """
    Plots predicted bounding box for given image

    :param image: Image to be plotted on
    :param boxes: List of bboxes
    :param class_labels: List of class labels
    :return: None
    """

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            (upper_left_y * height) - 5,
            s=class_labels[int(class_pred)],
            color=colors[int(class_pred)],
            verticalalignment="top",
            #bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def extract_bboxes(box_tensor: torch.Tensor, class_tensor: torch.Tensor,
                   anchors: torch.Tensor, stride: int,
                   num_classes: int = 4, num_anchors: int = 3,
                   conf_threshold: float = 0.3,
                   is_pred: bool = True,):
    """
    :param box_tensor:
    :param class_tensor:
    :param anchors: Scaled anchors as a tensor
    :param num_classes:
    :param num_anchors:
    :param conf_threshold:
    :param is_pred:
    :return:
    """

    if is_pred:
        box_tensor = box_tensor.permute(1, 2, 0)
        class_tensor = class_tensor.permute(1, 2, 0)

    box_tensor = box_tensor.view(*box_tensor.shape[:-1], num_anchors, 4).to("cpu")
    class_tensor = class_tensor.view(*class_tensor.shape[:-1], num_anchors, num_classes).to("cpu")

    preds = torch.where(torch.sigmoid(class_tensor) > conf_threshold)
    classes = []
    boxes = []
    for i in range(len(preds[0])):
        classes.append(preds[-1][i].item())
        class_proba = torch.sigmoid(class_tensor[preds[0][i], preds[1][i], preds[2][i], preds[-1][i]])
        box = box_tensor[preds[0][i], preds[1][i], preds[2][i], :]
        if is_pred:
            box[:2] = (torch.sigmoid(box[:2]) + torch.tensor((preds[0][i], preds[1][i]))) * stride
            box[2:] = ((torch.exp(box[2:])) * anchors[preds[2][i]]) * stride

        else:
            box[:2] = (box[:2] + torch.tensor((preds[0][i], preds[1][i]))) * stride
            box[2:] = (box[2:]) * stride

        box = box.tolist() + [class_proba.item()]
        boxes.append(box)

    return boxes, classes


def covert_yolo_to_bbox(bbox: torch.Tensor):
    upper_left = ((bbox[0] - bbox[2]).item(), (bbox[1] - bbox[3]).item())
    lower_right = ((bbox[0] + bbox[2]).item(), (bbox[1] + bbox[3]).item())

    return *upper_left, *lower_right


def denormalize(tensor):
    means_t = torch.tensor(config["MEANS"])
    stds_t = torch.tensor(config["STDS"])

    # Reshape mean and std to match the tensor dimensions for broadcasting
    means_t = means_t.view(1, -1, 1, 1)
    stds_t = stds_t.view(1, -1, 1, 1)

    # Denormalize
    denormalized_tensor = tensor * stds_t + means_t

    return denormalized_tensor


def plot_prediction(img: torch.Tensor, prediction: torch.Tensor,
                    scaled_anchor: torch.Tensor,
                    class_labels: list,
                    strides: list, threshold: float = 0.3,
                    num_classes: int = 4, is_pred: bool = True):

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    for i in range(prediction[0][0].shape[0]):
        fig, ax = plt.subplots()
        denorm_img = denormalize(img[i])
        ax.imshow(denorm_img[0].permute(1, 2, 0).cpu().detach())
        for s in range(len(strides)):
            out_box = prediction[s][1][i]
            out_class = prediction[s][0][i]
            boxes, classes = extract_bboxes(out_box, out_class, scaled_anchor[s], strides[s],
                                   num_classes, len(scaled_anchor[s]), threshold, is_pred)

            for box, cls in zip(boxes, classes):
                x, y, w, h = covert_yolo_to_patches(box[:4])
                rect = patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=colors[int(cls)],
                    facecolor="none",
                )
                ax.add_patch(rect)
                plt.text(
                    x, y -7,
                    s=class_labels[int(cls)],
                    #color=colors[int(cls)],
                    verticalalignment="top",
                    fontsize=14,
                    bbox={"color": colors[int(cls)], "pad": 0},
                )

        plt.show()

def covert_yolo_to_patches(bbox: torch.Tensor):
    xy = ((bbox[0] - (bbox[2] / 2)), (bbox[1] - (bbox[3] / 2)))

    return *xy, bbox[2], bbox[3]


def get_evaluation_bboxes(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    iou_threshold: float,
    anchors: List[float],
    threshold: float,
    box_format: str = "midpoint",
    device: str = "cuda",
):
    """
    Computes and returns all predicted and true bounding boxes for evaluation purposes.

    Args:
        loader (torch.utils.data.DataLoader): The data loader to iterate over the dataset.
        model (torch.nn.Module): The model to be evaluated.
        iou_threshold (float): The Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
        anchors (list): List of anchor box dimensions for each scale.
        threshold (float): The confidence threshold for filtering predicted bounding boxes.
        box_format (str, optional): The format of the bounding boxes. Defaults to "midpoint".
        device (str, optional): The device to which model and tensors should be moved before computation. Defaults to "cuda".

    Returns:
        list: A list containing all the predicted bounding boxes.
        list: A list containing all the true bounding boxes.
    """
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions: torch.Tensor, anchors: Union[List[float], torch.Tensor],
                    S: int, is_preds: bool = True):
    """
    Converts the bounding box predictions from the feature map/grid cells to actual bounding box coordinates.

    Args:
        predictions (torch.Tensor): The raw predictions from the model.
        anchors (list or torch.Tensor): The anchor boxes dimensions for each scale.
        S (int): The size of the feature map/grid cell.
        is_preds (bool, optional): Flag indicating whether the input are predictions or ground truth. Defaults to True.

    Returns:
        list: A list of converted bounding boxes in the format [best_class, score, x, y, width, height].

    Note:
        The function uses sigmoid and exponential operations for converting the box predictions and applies necessary
        normalization to convert the box coordinates relative to the input image dimensions.
    """

    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * (anchors) # * S ???
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()



def check_class_accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, threshold: float):
    """
    Args:
        model (torch.nn.Module): The model to be evaluated. It is expected
            to be an instance of a PyTorch model, typically an object detection model.
        loader (torch.utils.data.DataLoader): The DataLoader containing the
            dataset over which the accuracy is to be evaluated.
        threshold (float): The threshold for considering a prediction
            as positive. Any prediction above this threshold is considered positive.

    Returns:
        None. The computed accuracies are printed to the console.
    """
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config["DEVICE"])
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            target = y[i][:, 0].to(config["DEVICE"])
            pred_cleaned = torch.sigmoid(out[i][0]).permute(0, 2, 3, 1)
            pred_cleaned = (pred_cleaned > threshold).float()
            obj_mask = (target == 1)
            noobj_mask = (target != 1)


            correct_class += torch.sum(pred_cleaned[obj_mask] == target[obj_mask])
            tot_class_preds += torch.sum(obj_mask)

            correct_noobj += torch.sum(pred_cleaned[noobj_mask] == target[noobj_mask])
            tot_noobj += torch.sum(noobj_mask)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    model.train()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str = "my_checkpoint.path.tar"):
    print("=> saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_file: str, lr: float):
    print("Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config["DEVICE"])
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def plot_couple_examples(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                         thresh: float, iou_thresh: float, anchors: List[List[float]],
                         device: str, class_labels: List[str]):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    anchors = anchors.to(device)
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes, class_labels)



def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(dataset):

    strides = config["STRIDES"]
    img_size = config["IMG_SIZE"]
    scales = [img_size // strides[0], img_size // strides[1], img_size // strides[2]]

    train_dataset = dataset(config["TRAIN_DIR"] + "/labels",
                                config["TRAIN_DIR"] + "/images",
                                config["ANCHORS"],
                                img_size,
                                scales,
                                config["NUM_CLASSES"],
                                transforms=train_transforms)

    val_dataset = dataset(config["VAL_DIR"] + "/labels",
                              config["VAL_DIR"] + "/images",
                              config["ANCHORS"],
                              img_size,
                              scales,
                              config["NUM_CLASSES"],
                              transforms=test_transforms)

    test_dataset = dataset(config["TEST_DIR"] + "/labels",
                               config["TEST_DIR"] + "/images",
                               config["ANCHORS"],
                               img_size,
                               scales,
                               config["NUM_CLASSES"],
                               transforms=test_transforms)

    train_loader = DataLoader(train_dataset,
                               batch_size=config["BATCH_SIZE"],
                               num_workers=config["NUM_WORKERS"],
                               pin_memory=config["PIN_MEMORY"],
                               shuffle=True,
                               drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=config["BATCH_SIZE"],
                             num_workers=config["NUM_WORKERS"],
                             pin_memory=config["PIN_MEMORY"],
                             shuffle=False,
                             drop_last=False)

    val_loader = DataLoader(val_dataset,
                            batch_size=config["BATCH_SIZE"],
                            num_workers=config["NUM_WORKERS"],
                            pin_memory=config["PIN_MEMORY"],
                            shuffle=False,
                            drop_last=False)

    return train_loader, test_loader, val_loader

