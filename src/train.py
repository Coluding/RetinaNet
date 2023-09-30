import torch.cuda.amp
import torch
from tqdm import tqdm

from model import RetinaNet
from loss import RetinaLoss
from dataset import RetinaDataset
from utils import get_loaders, config, mean_average_precision, check_class_accuracy, save_checkpoint,load_checkpoint


def train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader)
    losses = []
    box_losses = []
    fc_losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config["DEVICE"])

        with torch.cuda.amp.autocast():
            out = model(x)
            loss0, fc_loss0, box_loss0 = loss_fn(out[0], y[0], scaled_anchors[0])
            loss1, fc_loss1, box_loss1 = loss_fn(out[1], y[1], scaled_anchors[1])
            loss2, fc_loss2, box_loss2 = loss_fn(out[2], y[2], scaled_anchors[2])

            loss = loss0 + loss1 + loss2
            fc_loss = fc_loss0 + fc_loss1 + fc_loss2
            box_loss = box_loss0 + box_loss1 + box_loss2
            box_losses.append(box_loss)
            losses.append(loss)
            fc_losses.append(fc_loss)

            losses.append(loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        mean_loss = sum(losses) / len(losses)
        mean_box_loss = sum(box_losses) / len(losses)
        mean_fc_loss = sum(fc_losses) / len(losses)
        loop.set_postfix(loss=mean_loss.item(), box_loss=mean_box_loss.item(), fc_loss=mean_fc_loss.item())


def val(val_loader, model, scaled_anchors, loss_fn):
    loop = tqdm(val_loader)
    losses = []
    box_losses = []
    fc_losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config["DEVICE"])
        out = model(x)
        loss0, fc_loss0, box_loss0 = loss_fn(out[0], y[0], scaled_anchors[0])
        loss1, fc_loss1, box_loss1 = loss_fn(out[1], y[1], scaled_anchors[1])
        loss2, fc_loss2, box_loss2 = loss_fn(out[2], y[2], scaled_anchors[2])

        loss = loss0 + loss1 + loss2
        fc_loss = fc_loss0 + fc_loss1 + fc_loss2
        box_loss = box_loss0 + box_loss1 + box_loss2
        box_losses.append(box_loss)
        losses.append(loss)
        fc_losses.append(fc_loss)

        mean_loss = sum(losses) / len(losses)
        mean_box_loss = sum(box_losses) / len(losses)
        mean_fc_loss = sum(fc_losses) / len(losses)
        loop.set_postfix(validation_loss=mean_loss.item(), validation_box_loss=mean_box_loss.item(),
                         validation_fc_loss=mean_fc_loss.item())



def main():
    import warnings
    warnings.simplefilter("ignore")
    anchors = config["ANCHORS"]
    model = RetinaNet(config["CHANNEL_SCALES"], config["NUM_CLASSES"], len(anchors[0]), config["TRAIN_BACKBONE"])
    model = model.to(config["DEVICE"])
    train_laoder, val_loader, test_loader = get_loaders(RetinaDataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["LEARNING_RATE"]),
                                           weight_decay=float(config["WEIGHT_DECAY"]))

    loss = RetinaLoss(config["NUM_CLASSES"], config["GAMMA"], config["ALPHA"], config["DEVICE"], config["W1LOSS"],
                      config["W2LOSS"])
    scaler = torch.cuda.amp.GradScaler()

    if config["LOAD_CHECKPOINT"]:
        load_checkpoint(model, optimizer, config["SAVE_PATH"], config["LEARNING_RATE"])

    scaled_anchors = (torch.tensor(anchors) * (1/torch.tensor(config["STRIDES"]) * (config["IMG_SIZE"]))
                      .unsqueeze(1)).to(config["DEVICE"])

    for epoch in range(config["EPOCHS"]):
        print(f"In epoch {epoch}")
        train(train_laoder, model, optimizer, loss, scaler, scaled_anchors)
        val(val_loader, model, scaled_anchors, loss)

        save_checkpoint(model, optimizer, config["SAVE_PATH"])


        if epoch % 10 == 0 and epoch > 0:
            check_class_accuracy(model, test_loader, threshold=config["CONF_THRESHOLD"])


if __name__ == "__main__":
    main()



