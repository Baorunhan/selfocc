import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from typing import Optional, Iterable, List, Union
import pickle
import os
import numpy as np
import cv2
from pyquaternion import Quaternion
import torchvision.transforms as transforms

def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


class PrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        if self.sequential:
            for adj_info in results['adjacent']:
                # adjacent与current使用相同的图像增广, 相机内参也相同.
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6

        bda_rot = torch.eye(3)

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda_rot

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results



class MultiModalDataset(Dataset):
    def __init__(self, pkl_path, img_transform=None):
        with open(pkl_path, 'rb') as f:
            data_pikle = pickle.load(f)
        
        self.data = data_pikle['infos']
        self.img_transform = img_transform
        self.text_tokenizer = None  # 可以添加文本tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data_input = {}
        data_input.update({'curr':item})
        if self.img_transform != None:
            img_inputs = self.img_transform(data_input)
        else:
            RuntimeError("img_transform is None!")
        # 图像数据
        occ_gt_path = os.path.join(item['occ_path'], "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_camera = occ_labels['mask_camera']
        
        return {
            'img_inputs': img_inputs['img_inputs'],
            'voxel_semantics': semantics,
            'mask_camera': mask_camera,
        }

if __name__ == '__main__': 
    
    data_config = {
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT'
        ],
        'Ncams':
        6,
        'input_size': (256, 704),
        'src_size': (900, 1600),

        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }   
    train_transform = transforms.Compose([PrepareImageInputs(data_config)])
    dataset = MultiModalDataset('./data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', img_transform=train_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        # collate_fn=multimodal_collate_fn,
        num_workers=0
    )

    for batch in dataloader:
        images = batch['img_inputs']
        voxel_semantics = batch['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = batch['mask_camera']
        
        # 训练代码...   
    
