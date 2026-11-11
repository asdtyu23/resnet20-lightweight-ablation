import torch
import torch.nn as nn

# 标准卷积
class StdConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=False) # 修改：关闭原地操作
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

# 深度可分离卷积
class DWConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=False) # 修改：关闭原地操作
    def forward(self, x): 
        out = self.relu(self.bn1(self.dw(x)))
        return self.relu(self.bn2(self.pw(out)))

# SE 注意力
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False), # 修改：关闭原地操作
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_dw=False, use_se=False):
        super().__init__()
        ConvLayer = DWConv if use_dw else StdConv
        self.conv1 = ConvLayer(in_c, out_c, stride)
        self.conv2 = ConvLayer(out_c, out_c, 1)
        self.se = SEBlock(out_c) if use_se else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + self.shortcut(x) # 修改：避免使用 += 原地操作
        return torch.relu(out)

class ResNet20(nn.Module):
    def __init__(self, num_classes=10, use_dw=False, use_se=False):
        super().__init__()
        self.in_c = 16
        self.conv1 = StdConv(3, 16)
        self.layer1 = self._make_layer(16, 3, 1, use_dw, use_se)
        self.layer2 = self._make_layer(32, 3, 2, use_dw, use_se)
        self.layer3 = self._make_layer(64, 3, 2, use_dw, use_se)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_c, num_blocks, stride, use_dw, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_c, out_c, s, use_dw, use_se))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)