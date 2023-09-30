import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import config, get_loaders, check_class_accuracy, extract_bboxes, covert_yolo_to_patches, plot_prediction
from model import RetinaNet
from dataset import RetinaDataset


def setup():
    global model
    global scaled_anchors
    global train_loader, val_loader, test_loader
    checkpoint = torch.load(config["SAVE_PATH"])
    anchors = torch.tensor(config["ANCHORS"])
    scaled_anchors = (torch.tensor(anchors) * (1 / torch.tensor(config["STRIDES"]) * (config["IMG_SIZE"]))
                      .unsqueeze(1))
    model = RetinaNet(config["CHANNEL_SCALES"], config["NUM_CLASSES"], len(anchors[0]), config["TRAIN_BACKBONE"]).to("cuda")
    model.load_state_dict(checkpoint["state_dict"])
    train_loader, val_loader, test_loader = get_loaders(RetinaDataset)


def main():
    import warnings
    warnings.simplefilter("ignore")
    setup()
    img, lab = next(iter(val_loader))
    out = model(img.to("cuda"))
    #out = lab
    plot_prediction(img, out, scaled_anchors, config["CLASS_LABELS"], config["STRIDES"],
                    config["CONF_THRESHOLD"], config["NUM_CLASSES"], True)

    #check_class_accuracy(model, train_loader, 0.3)


if __name__ == "__main__":
    main()

