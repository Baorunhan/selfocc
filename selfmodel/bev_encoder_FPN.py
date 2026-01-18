import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import BasicBlock


class bev_encoder_with_FPN(nn.Module):
    def __init__(
            self,
            numC_input,# backbone input
            in_channels,# FPN input
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            input_feature_index=[0, 2],
            extra_upsample=2,
            out_channels=256,
            block_type='Basic',
    ):
        super(bev_encoder_with_FPN, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BasicBlock(in_channels=curr_numC, out_channels=num_channels[i], stride=stride[i],
                                downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1))]
            curr_numC = num_channels[i]
            layer.extend([BasicBlock(in_channels=curr_numC, out_channels=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)
        
        ## bev FPN_LSS 部分
        self.extra_upsample = extra_upsample is not None
        self.input_feature_index = input_feature_index
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
        )
        

        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        
        # 使用最后两个特征进行BEV融合
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x       
        
    

