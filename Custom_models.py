from torch import nn as nn


# model
class REDCN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_conv = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 30, 5),
            nn.ReLU(),
            nn.Conv2d(30, 40, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
        )

        self.relu_linear = nn.Sequential(
            nn.Linear(360, 700),
            nn.SELU(),
            nn.Linear(700, 300),
            nn.SELU(),
            # nn.Dropout(p=0.25),
            nn.Linear(300, 150),
            nn.SELU(),
            nn.Linear(150, 80),
            nn.SELU(),
            nn.Linear(80, 100),
        )

    def forward(self, x):
        conv_layers = self.relu_conv(x)
        linear_layers = self.relu_linear(conv_layers)
        return linear_layers


# VGG
class vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
        )

        self.relu_linear = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 100),
        )

    def forward(self, x):
        conv_layers = self.relu_conv(x)
        linear_layers = self.relu_linear(conv_layers)
        return linear_layers


def conv(channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = conv(3, 150)
        self.conv2 = conv(150, 300, pool=True)
        self.res1 = nn.Sequential(conv(300, 300), conv(300, 300))

        self.conv3 = conv(300, 500, pool=True)
        self.conv4 = conv(500, 1000, pool=True)
        self.res2 = nn.Sequential(conv(1000, 1000), conv(1000, 1000))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(p=0.2), nn.Linear(1000, 100)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
