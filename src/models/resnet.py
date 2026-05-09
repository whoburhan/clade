import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        self.out_dim = 512

    def forward(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, num_classes=7, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class SplitModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, head_hidden_dim=128, head_dropout=0.3):
        super().__init__()
        self.extractor = FeatureExtractor(pretrained=pretrained)
        self.head = ClassificationHead(
            in_dim=self.extractor.out_dim, hidden_dim=head_hidden_dim,
            num_classes=num_classes, dropout=head_dropout,
        )

    def forward(self, x):
        return self.head(self.extractor(x))

    def get_extractor_params(self):
        return self.extractor.state_dict()

    def get_head_params(self):
        return self.head.state_dict()

    def set_extractor_params(self, state_dict):
        self.extractor.load_state_dict(state_dict)

    def set_head_params(self, state_dict):
        self.head.load_state_dict(state_dict)


def create_model(num_classes=7, pretrained=True, head_hidden_dim=128, head_dropout=0.3):
    return SplitModel(num_classes=num_classes, pretrained=pretrained,
                      head_hidden_dim=head_hidden_dim, head_dropout=head_dropout)
