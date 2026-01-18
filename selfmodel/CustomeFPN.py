import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomFPN(nn.Module):
    r"""Feature Pyramid Network (FPN) 纯 PyTorch 实现
    保留原 MMCV 版本的核心逻辑：
    - 可配置输入通道、输出通道、输出层级数
    - 支持 top-down 路径的上采样融合
    - 可配置额外的卷积层（Extra Convs）
    - 可指定输出的层级 ID
    """

    def __init__(self,
                 in_channels=[256,512],
                 out_channels=512,
                 num_outs=0,
                 start_level=0,
                 end_level=-1,
                 out_ids=[],
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,  # 兼容原参数，实际仅支持默认卷积
                 norm_cfg=None,  # 仅支持 BatchNorm2d (dict(type='BN')) 或 None
                 act_cfg=None,   # 支持 'relu'/'gelu'/'silu' 或 None
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=None):  # 兼容原参数，初始化逻辑手动实现
        super(CustomFPN, self).__init__()
        assert isinstance(in_channels, list), "in_channels 必须是列表"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.out_ids = out_ids

        # 处理 end_level 逻辑
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels), "end_level 超出输入通道列表长度"
            assert num_outs == end_level - start_level, "num_outs 与层级范围不匹配"
        self.start_level = start_level
        self.end_level = end_level

        # 处理 add_extra_convs 逻辑
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output'), \
                "add_extra_convs 仅支持 'on_input'/'on_lateral'/'on_output'"
        elif add_extra_convs:  # True 等价于 'on_input'
            self.add_extra_convs = 'on_input'

        # 解析归一化和激活配置
        self.norm_layer = self._build_norm_layer(norm_cfg)
        self.act_layer = self._build_act_layer(act_cfg)

        # 构建 lateral 卷积（1x1 降维）和 fpn 卷积（3x3 融合）
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # 构建 lateral 卷积（1x1）
            lateral_conv = self._build_conv_block(
                in_channels=self.in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                use_norm=not self.no_norm_on_lateral
            )
            self.lateral_convs.append(lateral_conv)

            # 仅为指定的 out_ids 构建 fpn 卷积（3x3）
            if i in self.out_ids:
                fpn_conv = self._build_conv_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    use_norm=True
                )
                self.fpn_convs.append(fpn_conv)

        # 构建额外的卷积层（如 RetinaNet 所需的额外层级）
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_ch = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_ch = out_channels
                extra_conv = self._build_conv_block(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    use_norm=True
                )
                self.fpn_convs.append(extra_conv)

        # 初始化权重（模拟原 init_cfg 的 Xavier 初始化）
        self._initialize_weights()

    def _build_norm_layer(self, norm_cfg):
        """构建归一化层（仅支持 BatchNorm2d 或 None）"""
        if norm_cfg is None:
            return None
        if isinstance(norm_cfg, dict) and norm_cfg.get('type') == 'BN':
            return nn.BatchNorm2d
        raise NotImplementedError(f"仅支持 BN 归一化，不支持 {norm_cfg}")

    def _build_act_layer(self, act_cfg):
        """构建激活层（支持 relu/gelu/silu 或 None）"""
        if act_cfg is None:
            return None
        if isinstance(act_cfg, dict):
            act_type = act_cfg.get('type', 'relu').lower()
        elif isinstance(act_cfg, str):
            act_type = act_cfg.lower()
        else:
            raise TypeError("act_cfg 必须是字典或字符串")

        if act_type == 'relu':
            return nn.ReLU(inplace=False)
        elif act_type == 'gelu':
            return nn.GELU()
        elif act_type == 'silu':
            return nn.SiLU(inplace=False)
        raise NotImplementedError(f"不支持的激活函数 {act_type}")

    def _build_conv_block(self, in_channels, out_channels, kernel_size, 
                          stride=1, padding=0, use_norm=True):
        """构建卷积块：Conv2d + (BN) + (Activation)"""
        layers = []
        # 卷积层
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_norm  # 有BN时bias设为False
        )
        layers.append(conv)

        # 归一化层
        if use_norm and self.norm_layer is not None:
            bn = self.norm_layer(out_channels)
            layers.append(bn)

        # 激活层
        if self.act_layer is not None:
            layers.append(self.act_layer)

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化（Xavier 均匀分布）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """前向传播函数（与原逻辑完全一致）"""
        assert len(inputs) == len(self.in_channels), \
            f"输入特征数 {len(inputs)} 与 in_channels 长度 {len(self.in_channels)} 不匹配"

        # 1. 构建 lateral 特征（1x1 卷积降维）
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 构建 top-down 路径（上采样 + 逐元素相加）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                # 按缩放因子上采样
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                # 按目标尺寸上采样（匹配前一层的空间尺寸）
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        # 3. 构建基础输出（仅 out_ids 指定的层级）
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]

        # 4. 补充额外的输出层级（若 num_outs > 基础输出数）
        if self.num_outs > len(outs):
            # 情况1：不添加额外卷积，用最大池化生成（如 Faster R-CNN）
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
            # 情况2：添加额外卷积层（如 RetinaNet）
            else:
                # 确定额外卷积的输入源
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                # 第一个额外卷积
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                # 后续额外卷积
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return outs

def CustomeFPN_V1( kwargs) -> CustomFPN:
    if 'neck' in kwargs:
        neck_config = kwargs['neck']
        return CustomFPN(neck_config.get('in_channels', [256, 512]),
        out_channels=neck_config.get('out_channels', 256),
        num_outs=neck_config.get('num_outs', 1),
        start_level=0,
        end_level=-1,
        out_ids=[0],
        add_extra_convs=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='relu'),
        upsample_cfg=dict(mode='nearest'))
    else:
        return CustomFPN(in_channels=[256, 512],
        out_channels=256,
        num_outs=1,
        start_level=0,
        end_level=-1,
        out_ids=[0],
        add_extra_convs=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='relu'),
        upsample_cfg=dict(mode='nearest'))

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 模拟输入：4个层级的特征图（对应 ResNet 的 C1-C4）
    in_channels = [256, 512]
    # scales = [224, 112, 56, 28]  # 各层级的空间尺寸
    inputs = [torch.randn(6, 256, 16,44) , torch.randn(6, 512, 8, 22)] 
    input_dummy = tuple(inputs)
    # 初始化 FPN
    fpn = CustomFPN(
        in_channels=in_channels,
        out_channels=256,
        num_outs=1,
        start_level=0,
        end_level=-1,
        out_ids=[0],
        add_extra_convs=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='relu'),
        upsample_cfg=dict(mode='nearest')
    )

    # 前向传播
    fpn.eval()
    with torch.no_grad():
        outputs = fpn(inputs)

    # 打印输出形状（验证是否符合预期）
    for i, out in enumerate(outputs):
        print(f"outputs[{i}].shape = {out.shape}")
    # 预期输出：
    # outputs[0].shape = torch.Size([1, 256, 224, 224])
    # outputs[1].shape = torch.Size([1, 256, 112, 112])
    # outputs[2].shape = torch.Size([1, 256, 56, 56])
    # outputs[3].shape = torch.Size([1, 256, 28, 28])
