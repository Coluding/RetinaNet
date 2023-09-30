import torch
import torch.nn as nn
import torchvision


class ClassPredictBlock(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 4, num_anchors: int = 3):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
            nn.ReLU()
        ) for _ in range(num_convs)])

        self.prediction_layer = nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=3, padding="same")

    def forward(self, x):
        features = self.convs(x)
        return self.prediction_layer(features)


class BoxPredictionBlock(nn.Module):
    def __init__(self, in_channels: int, num_convs: int = 4, num_anchors: int = 3):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
            nn.ReLU()
        ) for _ in range(num_convs)])

        self.prediction_layer = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=3, padding="same")

    def forward(self, x):
        features = self.convs(x)
        return self.prediction_layer(features)


class RetinaNet(nn.Module):
    def __init__(self,
                 channel_scales: list = [2048, 1024, 512],
                 num_classes: int = 4,
                 num_anchors: int = 3,
                 train_backbone: bool = False):
        super().__init__()
        backbone = torchvision.models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        for param in self.backbone.parameters():
            param.requires_grad = train_backbone

        self.outputs = {} # I will use a dict for debugging reasons, otherwise a list would be sufficient

        def hook(module, input, output):
            self.outputs[module] = output

        for layer in list(self.backbone.children())[-3:]:
            layer.register_forward_hook(hook)

        self.class_prediction_blocks = nn.ModuleList([ClassPredictBlock(ch, num_classes, 4, num_anchors)
                                                      for ch in channel_scales])

        self.box_prediction_blocks = nn.ModuleList([BoxPredictionBlock(ch, num_classes, 3)
                                                    for ch in channel_scales])

        self.upsample_1_to_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                             nn.Conv2d(channel_scales[0], channel_scales[1], kernel_size=1))
        self.upsample_2_to_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                             nn.Conv2d(channel_scales[1], channel_scales[2], kernel_size=1))


    def forward(self, x):
        self.backbone(x)

        backbone_outputs = list(self.outputs.values())
        scale_1_output = backbone_outputs[-1]
        scale_2_output = backbone_outputs[-2]
        scale_3_output = backbone_outputs[-3]

        predictions = []

        prediction_1_class = self.class_prediction_blocks[0](scale_1_output)
        prediction_1_box = self.box_prediction_blocks[0](scale_1_output)
        predictions.append([prediction_1_class, prediction_1_box])

        residual_input = self.upsample_1_to_2(scale_1_output) + scale_2_output
        prediction_2_class = self.class_prediction_blocks[1](residual_input)
        prediction_2_box = self.box_prediction_blocks[1](residual_input)
        predictions.append([prediction_2_class, prediction_2_box])

        residual_input = self.upsample_2_to_3(scale_2_output) + scale_3_output
        prediction_3_class = self.class_prediction_blocks[2](residual_input)
        prediction_3_box = self.box_prediction_blocks[2](residual_input)
        predictions.append([prediction_3_class, prediction_3_box])

        return predictions



