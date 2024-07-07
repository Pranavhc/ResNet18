import torch 
import torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    def __init__(self, in_chnls:int, out_chnls:int, stride:int=1, downsample:bool=False) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chnls, out_chnls, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chnls)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chnls, out_chnls, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chnls)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chnls, out_chnls, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chnls)
            )

    def forward(self, x):
        residual = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None: 
            residual = self.downsample(residual)

        x += residual
        out = self.relu(x)

        return out
    
class ResNet18(nn.Module):
    def __init__(self, n_classes:int=10) -> None:
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   # img 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                     # 16x16 -> 8x8

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),                                                             # 8x8 -> 8x8
            BasicBlock(64, 64)                                                              # 8x8 -> 8x8
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),                                 # 8x8 -> 4x4
            BasicBlock(128, 128)                                                            # 4x4 -> 4x4
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),                                # 4x4 -> 2x2
            BasicBlock(256, 256)                                                            # 2x2 -> 2x2
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),                                # 2x2 -> 1x1
            BasicBlock(512, 512)                                                            # 1x1 -> 1x1
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # initial downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)     # (b, c, 1, 1)
        x = torch.flatten(x, 1) # (b, c)
        x = self.fc(x)

        return x
    
    @classmethod
    def from_pretrained(self, n_classes:int=10) -> nn.Module:
        print('Loading weights for ResNet18 model')

        # our model
        model = ResNet18(n_classes)
        r18 = model.state_dict()
        r18_keys = r18.keys()

        # pretrained model
        p_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        p_r18 = p_model.state_dict()
        p_r18_keys = p_r18.keys()

        assert len(r18_keys) == len(p_r18_keys), f"missmatched number of keys: {len(r18_keys)} != {len(p_r18_keys)}"

        # load pretrained weights into our model
        for k in p_r18_keys:
            if k.startswith('fc'): continue
            with torch.no_grad():
                assert r18[k].shape == p_r18[k].shape, f"missmatched shapes: {r18[k].shape} != {p_r18[k].shape}"
                r18[k].copy_(p_r18[k])
        
        return model