import torch.nn as nn
import torch
from typing import Optional, List, Type, Union

# 定义基础残差块（用于ResNet18/34）
class BasicBlock(nn.Module):
    expansion: int = 1  # 通道扩展倍数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: Type[nn.Module] = nn.ReLU,  # 可配置激活函数
        norm_layer: Type[nn.Module] = nn.BatchNorm2d  # 可配置归一化层
    ) -> None:
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = activation(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        # 下采样（用于匹配维度）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：如果需要下采样则先下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义瓶颈块（用于ResNet50/101/152）
class Bottleneck(nn.Module):
    expansion: int = 4  # 通道扩展倍数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d
    ) -> None:
        super().__init__()
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        
        self.relu = activation(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



# 可配置的ResNet主类
class ConfigurableResNet(nn.Module):
    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],  # 块类型（BasicBlock/Bottleneck）
        layers: List[int],  # 每阶段的块数，如[2,2,2,2]对应ResNet18
        in_channels: int = 3,  # 输入通道数（RGB为3，灰度图为1）
        base_channels: int = 64,  # 基础通道数
        activation: Type[nn.Module] = nn.ReLU,  # 激活函数
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,  # 归一化层
        use_maxpool: bool = True,  # 是否使用初始最大池化
        dropout_rate: float = 0.0  # dropout概率（0则不使用）
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        self.activation = activation
        self.in_planes = base_channels  # 当前输入通道数
        layers_num = len(layers)
        self.layers_name_list = []
        for i in range(layers_num):
            self.layers_name_list.append(f'layer{i+1}')
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = activation(inplace=True)
        
        # 初始最大池化（可配置）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity()

        # 构建4个阶段的残差块
        self.layers = []
        for i in range(layers_num):
            layer = self._make_layer(
                block,
                base_channels * (2 ** i),
                layers[i],
                stride=1 if i == 0 else 2  # 第一个阶段不下采样，后续阶段下采样
            )
            setattr(self, f'layer{i+1}', layer)
            self.layers.append(layer)
              
        # self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """构建单个阶段的残差块序列"""
        downsample = None
        # 需要下采样的情况：步长>1 或 输入通道≠输出通道
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.in_planes, planes, stride, downsample, self.activation, self.norm_layer))
        self.in_planes = planes * block.expansion
        # 后续块不需要下采样
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, activation=self.activation, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """初始化卷积和批归一化层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        
        # x = x.view(-1, *x.shape[2:])  # 展平批次和视图维x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.layers_name_list):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in [2,3]:  # 返回layer1, layer2, layer3的输出
                outs.append(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return tuple(outs)


def resnet18(kwargs) -> ConfigurableResNet:
    if 'backbone' in kwargs:
        backbone_config = kwargs['backbone']
        return ConfigurableResNet(BasicBlock, layers=backbone_config.get('layers', [2, 2, 2, 2]),
                                  in_channels=backbone_config.get('in_channels', 3),
                                  base_channels=backbone_config.get('base_channels', 64))
    else:
        return ConfigurableResNet(BasicBlock, [2, 2, 2, 2], **kwargs)



# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 模拟输入：(1, 2, 3, 256 704)

    # 初始化 FPN
    inputs = torch.randn(1, 6, 3, 256, 704)
    model = resnet18()

    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    # 打印输出形状（验证是否符合预期）
    for i, out in enumerate(outputs):
        print(f"outputs[{i}].shape = {out.shape}")
    # 预期输出：
    # outputs[0].shape = torch.Size([1, 256, 224, 224])
    # outputs[1].shape = torch.Size([1, 256, 112, 112])
    # outputs[2].shape = torch.Size([1, 256, 56, 56])
    # outputs[3].shape = torch.Size([1, 256, 28, 28])

