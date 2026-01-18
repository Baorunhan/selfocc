import torch.nn as nn
import torch
from typing import Optional, List, Type
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, training, backbone, neck, img_view_transformer, img_bev_encoder, occ_head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.img_view_transformer = img_view_transformer
        self.img_bev_encoder = img_bev_encoder
        self.occ_head = occ_head
        self.training = training
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.backbone(imgs)
        if self.neck != None:
            x = self.neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x 
        
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
        inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)



        # global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)  # brh
        global2keyego = torch.tensor([[[[ 8.7724e-01, -4.7973e-01, -1.7649e-02,  2.6427e+02],
          [ 4.7987e-01,  8.7733e-01,  4.8901e-03, -1.7334e+03],
          [ 1.3138e-02, -1.2759e-02,  9.9983e-01,  1.3143e+01],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]],
           device='cuda:0', dtype=torch.float64)

        sensor2keyegos = \
        global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
            post_rots, post_trans, bda]
        
    def forward(self, img_inputs, img_metas, points, **kwargs):
        if self.training:
            return self.forward_train(img_inputs, img_metas, points, **kwargs)
        else:
            return self.forward_test(img_inputs, img_metas, points)
            
    def forward_test(self, img_inputs, img_metas, points):
        img_inputs = self.prepare_inputs(img_inputs)
        x = self.image_encoder(img_inputs[0])
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # img_feats = self.bev_encoder(x)
        # occ_bev_feature = img_feats
        # occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        
        return x


    def forward_train(self,
                      img_inputs=None,
                      img_metas=None,
                      points=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_inputs = self.prepare_inputs(img_inputs)
        x = self.image_encoder(img_inputs[0])
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        img_feats = self.bev_encoder(x)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats
        # if self.upsample:
        #     occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
        #                                     mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses


    
    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_preds = self.occ_head(img_feats)
        # if not hasattr(self.occ_head, "get_occ_gpu"):
        #     occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # else:
        #     occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds