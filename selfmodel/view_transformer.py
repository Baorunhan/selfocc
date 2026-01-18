import torch
import torch.nn as nn

class LSSViewTransformer(nn.Module):
    def __init__(self, grid_config, out_channels, in_channels, downsample=16, \
                 collapse_z=False, input_size = (256, 704) ):
        super().__init__()
        # 初始化代码
        self.frustum = self.create_frustum([1.0, 45.0, 0.5],
                                           input_size, downsample) 
        self.grid_config = grid_config
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.downsample = downsample
        self.collapse_z = collapse_z
        self.input_size = input_size
        self.create_grid_infos(**grid_config)

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])     # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])        # (dx, dy, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])                   # (Dx, Dy, Dz)

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        # if self.sid:
        #     d_sid = torch.arange(self.D).float()
        #     depth_cfg_t = torch.tensor(depth_cfg).float()
        #     d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
        #                       torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
        #     d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)


    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                        bda):
            """Calculate the locations of the frustum points in the lidar
            coordinate system.

            Args:
                sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                    ego coordinate system in shape (B, N_cams, 4, 4).
                ego2global (torch.Tensor): Translation from ego coordinate system to
                    global coordinate system in shape (B, N_cams, 4, 4).
                cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                    (B, N_cams, 3, 3).
                post_rots (torch.Tensor): Rotation in camera coordinate system in
                    shape (B, N_cams, 3, 3). It is derived from the image view
                    augmentation.
                post_trans (torch.Tensor): Translation in camera coordinate system
                    derived from image view augmentation in shape (B, N_cams, 3).
                bda (torch.Tensor): Transformation in bev. (B, 3, 3)

            Returns:
                torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
            """
            B, N, _, _ = sensor2ego.shape

            # post-transformation
            # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
            points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
            # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
            
            post_rots_inv = torch.tensor([[2.2727, 0.0000, 0.0000],
            [0.0000, 2.2727, 0.0000],
            [0.0000, 0.0000, 1.0000]]).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, 3, 3)
            post_rots_inv_tensor = post_rots_inv.repeat(1, 6, 1, 1)
            
            
            # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            #     .matmul(points.unsqueeze(-1))  # brh
            points = post_rots_inv_tensor.view(B, N, 1, 1, 1, 3, 3)\
                .matmul(points.unsqueeze(-1))
            # cam_to_ego
            # (B, N_, D, fH, fW, 3, 1)  3: (du, dv, d)
            points = torch.cat(
                (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
            # R_{c->e} @ K^-1
            cam2imgs_inv_tensor = torch.tensor([[[[ 7.9500e-04,  0.0000e+00, -6.5766e-01],
            [ 0.0000e+00,  7.9500e-04, -3.5848e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

            [[ 7.9820e-04,  0.0000e+00, -6.5979e-01],
            [ 0.0000e+00,  7.9820e-04, -3.7514e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

            [[ 7.9570e-04,  0.0000e+00, -6.5072e-01],
            [ 0.0000e+00,  7.9570e-04, -3.5962e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

            [[ 7.9682e-04,  0.0000e+00, -6.6102e-01],
            [ 0.0000e+00,  7.9682e-04, -3.7225e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

            [[ 1.2549e-03,  0.0000e+00, -1.0764e+00],
            [ 0.0000e+00,  1.2549e-03, -5.9843e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

            [[ 8.0002e-04,  0.0000e+00, -6.6032e-01],
            [ 0.0000e+00,  8.0002e-04, -3.7005e-01],
            [ 0.0000e+00,  0.0000e+00,  1.0000e+00]]]]).cuda()  # (1, 6, 3, 3)
            # combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs)) # brh
            combine = sensor2ego[:, :, :3, :3].matmul((cam2imgs_inv_tensor))
            # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
            # --> (B, N, D, fH, fW, 3)
            points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
            points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)

            # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
            # --> (B, N, D, fH, fW, 3)
            points = bda.view(B, 1, 1, 1, 1, 3,
                            3).matmul(points.unsqueeze(-1)).squeeze(-1)
            return points

    def bev_pool_v2(self, depth, feat, ranks_depth, ranks_feat, ranks_bev,
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
        
    def voxel_pooling_v2(self, coor, depth, feat):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)     # (B, C, Dz, Dy, Dx)
            dummy = torch.cat(dummy.unbind(dim=2), 1)   # (B, C*Dz, Dy, Dx)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])       # (B, Dz, Dy, Dx, C)
        bev_feat = self.bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)    # (B, C, Dz, Dy, Dx)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)     # (B, C*Dz, Dy, Dx)
        return bev_feat
    
    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        
        target_1d_len = B * N * H * W
        # 修复2：显式校验维度合法性，避免shape不匹配
        assert num_points // D == target_1d_len, f"维度不匹配: {num_points//D} != {target_1d_len}"
        
        ranks_feat = torch.arange(
            0, num_points // D , dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        target_shape = (B, N, 1, H, W)
        ranks_feat = ranks_feat.reshape(*target_shape)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        # order = ranks_bev.argsort()
        sorted_values, order = torch.sort(ranks_bev)
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

        
    def view_transform_core(self, input, depth, tran_feat):
            """
            Args:
                input (list(torch.tensor)):
                    imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                    sensor2egos: (B, N, 4, 4)
                    ego2globals: (B, N, 4, 4)
                    intrins:     (B, N, 3, 3)
                    post_rots:   (B, N, 3, 3)
                    post_trans:  (B, N, 3)
                    bda_rot:  (B, 3, 3)
                depth:  (B*N, D, fH, fW)
                tran_feat: (B*N, C, fH, fW)
            Returns:
                bev_feat: (B, C*Dz(=1), Dy, Dx)
                depth: (B*N, D, fH, fW)
            """
            B, N, C, H, W = input[0].shape
            coor = self.get_ego_coor(*input[1:7])   # (B, N, D, fH, fW, 3)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))      # (B, C*Dz(=1), Dy, Dx)
            return bev_feat, depth
        
        
    def forward(self, input):
        # 前向传播代码
        x = input[0]    # (B, N, C_in, fH, fW)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x) 
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW)
        depth = depth_digit.softmax(dim=1) 
        return self.view_transform_core(input, depth, tran_feat)