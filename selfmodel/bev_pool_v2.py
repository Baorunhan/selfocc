import torch
import torch.nn.functional as F


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """
    纯 PyTorch 实现 BEV Pooling V2（无 CUDA 自定义算子依赖）
    Args:
        depth: (B, N, D, fH, fW) 深度特征，B=批次，N=相机数，D=深度分箱数，fH/fW=特征图高/宽
        feat:  (B, N, fH, fW, C) 图像特征，C=特征通道数
        ranks_depth: (N_points, ) 每个BEV点对应的depth张量展平索引
        ranks_feat:  (N_points, ) 每个BEV点对应的feat张量展平索引
        ranks_bev:   (N_points, ) 每个BEV点对应的BEV特征展平索引
        bev_feat_shape: (B, D_Z, D_Y, D_X, C) BEV特征形状，D_Z/D_Y/D_X=BEV栅格的Z/Y/X维度
        interval_starts: (N_pillar, ) 每个BEV栅格对应的特征点区间起始位置
        interval_lengths: (N_pillar, ) 每个BEV栅格对应的特征点数量

    Returns:
        x: bev feature in shape (B, C, D_Z, D_Y, D_X)
    """
    # -------------------------- 1. 解析基础参数 --------------------------
    B, D_Z, D_Y, D_X, C = bev_feat_shape
    N_points = ranks_depth.shape[0]
    N_pillar = interval_starts.shape[0]
    device = depth.device
    dtype = feat.dtype

    # -------------------------- 2. 展平深度和图像特征 --------------------------
    # 展平depth: (B, N, D, fH, fW) → (B*N*D*fH*fW, )
    depth_flat = depth.flatten()
    # 展平feat: (B, N, fH, fW, C) → (B*N*fH*fW, C)
    feat_flat = feat.flatten(0, 3)  # 前4维展平，保留通道维

    # -------------------------- 3. 提取每个BEV点对应的深度和特征 --------------------------
    # 提取深度值: (N_points, )
    depth_points = depth_flat[ranks_depth.long()]  # 转为long避免索引错误
    # 提取图像特征: (N_points, C)
    feat_points = feat_flat[ranks_feat.long()]

    # -------------------------- 4. 深度加权特征 --------------------------
    # 深度作为权重，加权图像特征 (N_points, C)
    weighted_feat = depth_points.unsqueeze(-1) * feat_points  # 扩展维度后逐通道加权

    # -------------------------- 5. 按BEV栅格区间累加特征 --------------------------
    # 初始化BEV特征为0: (B*D_Z*D_Y*D_X, C)
    bev_feat_flat = torch.zeros((B * D_Z * D_Y * D_X, C), device=device, dtype=dtype)

    # 方法1：利用scatter_add按区间累加（高效版）
    # 先对所有点按ranks_bev分组求和
    bev_feat_flat.scatter_add_(dim=0, index=ranks_bev.long().unsqueeze(-1).expand(-1, C), src=weighted_feat)

    # （可选）方法2：循环遍历每个pillar（易理解但速度较慢，适合调试）
    # for i in range(N_pillar):
    #     start = interval_starts[i]
    #     length = interval_lengths[i]
    #     if length == 0:
    #         continue
    #     end = start + length
    #     # 提取当前pillar的所有加权特征并求和
    #     pillar_feat = weighted_feat[start:end].sum(dim=0)
    #     # 填充到BEV特征中（ranks_bev[start]是该pillar对应的BEV索引）
    #     bev_feat_flat[ranks_bev[start]] = pillar_feat

    # -------------------------- 6. 恢复BEV特征形状并调整维度 --------------------------
    # 从展平形状恢复为 (B, D_Z, D_Y, D_X, C)
    bev_feat = bev_feat_flat.view(B, D_Z, D_Y, D_X, C)
    # 调整维度顺序为 (B, C, D_Z, D_Y, D_X)（符合深度学习框架的通道优先习惯）
    bev_feat = bev_feat.permute(0, 4, 1, 2, 3).contiguous()

    return bev_feat


# -------------------------- 测试代码（验证功能正确性） --------------------------
if __name__ == "__main__":
    # 模拟输入参数
    B, N, D, fH, fW, C = 2, 6, 16, 20, 20, 64  # 批次=2，相机=6，深度分箱=16，特征图20×20，通道=64
    D_Z, D_Y, D_X = 8, 80, 80  # BEV栅格维度 Z=8, Y=160, X=160
    N_points = 100000  # 总BEV点数量
    N_pillar = D_Z * D_Y * D_X * B  # BEV栅格总数

    # 生成模拟数据
    depth = torch.randn(B, N, D, fH, fW, device="cpu")
    feat = torch.randn(B, N, fH, fW, C, device="cpu")
    ranks_depth = torch.randint(0, B*N*D*fH*fW, (N_points,), device="cpu")
    ranks_feat = torch.randint(0, B*N*fH*fW, (N_points,), device="cpu")
    ranks_bev = torch.randint(0, B*D_Z*D_Y*D_X, (N_points,), device="cpu")
    bev_feat_shape = (B, D_Z, D_Y, D_X, C)
    # interval_starts_tmp = torch.randint(0, N_pillar, (N_points,), device="cpu")
    interval_starts,_ = torch.sort(torch.randint(0, N_pillar, (N_points,), device="cpu"))
    
    # interval_starts = torch.arange(0, N_points, N_points//N_pillar, device="cpu")[:N_pillar]
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    # 处理最后一个区间的长度（避免越界）
    # interval_lengths[-1] = N_points - interval_starts[-1]

    # 运行BEV Pooling
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           bev_feat_shape, interval_starts, interval_lengths)

    # 验证输出形状
    print(f"输入depth形状: {depth.shape}")
    print(f"输入feat形状: {feat.shape}")
    print(f"输出BEV特征形状: {bev_feat.shape}")  # 预期: (2, 64, 8, 200, 200)
