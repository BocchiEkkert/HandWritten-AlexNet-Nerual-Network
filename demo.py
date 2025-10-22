import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,config):
        super(AlexNet,self).__init__()
        self._config = config
        # 卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接层
        self.classifier = nn.Sequential(
            torch.flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self._config['num_classes']),
        )

        def forward(self,x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            return x
        