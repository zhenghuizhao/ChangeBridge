
from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils_cd import get_image_triplets_from_list
import cv2
import random
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import torchvision.transforms.functional as F
from pathlib import Path
import random
import json


try:
    # 兼容不同版本 torchvision
    from torchvision.transforms import InterpolationMode
except Exception:
    from torchvision.transforms.functional import InterpolationMode


class SyncTransform:
    """
    随机多尺度同步增强（纯几何，无颜色抖动/模糊/噪声）
    - 背景=黑色，mask 视为黑白：黑=背景(0), 白=前景(1)
    - 训练：随机 s∈[scale_min,scale_max] 放大 → （可选）连通域完整裁剪：裁剪窗四边必须为背景
           → 随机 0/90/180/270° → 水平/垂直翻转
    - 验证/测试：仅 Resize 到 image_size
    - 图像：BILINEAR；mask：NEAREST（保持硬边）
    - 兼容旧参数 is_scaled：若给 True，相当于 scale_range=(2.0,2.0)；False→(1.0,1.0)
    """
    def __init__(
        self,
        image_size,                  # (H, W)
        stage='train',
        is_scaled=False,             # 兼容旧接口；若提供 scale_range 将忽略此项
        scale_range=None,            # (min,max)，推荐下界>=1.0；None 时按 is_scaled 推断
        p_rot90=0.5,
        p_hflip=0.5,
        p_vflip=0.25,
        component_safe=False,        # 是否启用“连通域完整”裁剪（四边无前景）
        component_retries=16,        # 为找到满足条件的窗口最多尝试次数
        ensure_fg=False              # 是否要求裁剪窗内必须包含前景
    ):
        self.image_size  = image_size
        self.stage       = stage
        # 兼容：未显式给 scale_range 时，根据 is_scaled 设置
        if scale_range is None:
            scale_range = (2.0, 2.0) if is_scaled else (1.0, 1.0)
        self.scale_range = scale_range

        self.p_rot90     = p_rot90
        self.p_hflip     = p_hflip
        self.p_vflip     = p_vflip

        self.component_safe    = component_safe
        self.component_retries = component_retries
        self.ensure_fg         = ensure_fg

    def __call__(self, image_A: Image.Image, image_B: Image.Image, mask: Image.Image):
        # PIL -> Tensor in [0,1]
        A = TF.to_tensor(image_A)  # [3,H,W]
        B = TF.to_tensor(image_B)  # [3,H,W]
        M = TF.to_tensor(mask)     # [3,H,W]（黑白/三通道皆可）

        if self.stage == 'train':
            A, B, M = self._train(A, B, M)
        else:
            A, B, M = self._eval(A, B, M)
        return A, B, M

    # ------------------ 训练增强 ------------------
    def _train(self, A, B, M):
        Ht, Wt = self.image_size

        # 1) 随机多尺度（只放大到 >= 目标尺寸，便于裁剪回）
        s_min, s_max = self.scale_range
        s = random.uniform(s_min, s_max)
        new_h, new_w = max(int(Ht * s), Ht), max(int(Wt * s), Wt)

        A = TF.resize(A, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        B = TF.resize(B, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        M = TF.resize(M, (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        # 2) 裁剪：优先“连通域完整”（四边无前景），否则普通随机裁剪
        if self.component_safe and (new_h > Ht or new_w > Wt):
            i, j = self._sample_component_safe_coords(M, Ht, Wt)
            h, w = Ht, Wt
        else:
            i, j, h, w = transforms.RandomCrop.get_params(A, output_size=(Ht, Wt))

        A = TF.crop(A, i, j, h, w)
        B = TF.crop(B, i, j, h, w)
        M = TF.crop(M, i, j, h, w)

        # 3) 随机 0/90/180/270°
        if random.random() < self.p_rot90:
            k = random.choice([0, 1, 2, 3])
            if k:
                angle = 90 * k
                A = TF.rotate(A, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0.0)
                B = TF.rotate(B, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0.0)
                M = TF.rotate(M, angle, interpolation=InterpolationMode.NEAREST,  expand=False, fill=0.0)

        # 4) 水平/垂直翻转
        if random.random() < self.p_hflip:
            A = TF.hflip(A); B = TF.hflip(B); M = TF.hflip(M)
        if random.random() < self.p_vflip:
            A = TF.vflip(A); B = TF.vflip(B); M = TF.vflip(M)

        return A, B, M

    # ------------------ 验证/测试 ------------------
    def _eval(self, A, B, M):
        Ht, Wt = self.image_size
        A = TF.resize(A, (Ht, Wt), interpolation=InterpolationMode.BILINEAR)
        B = TF.resize(B, (Ht, Wt), interpolation=InterpolationMode.BILINEAR)
        M = TF.resize(M, (Ht, Wt), interpolation=InterpolationMode.NEAREST)
        return A, B, M

    # ------------------ 连通域完整裁剪（四边无前景=黑以外） ------------------
    def _sample_component_safe_coords(self, M: torch.Tensor, out_h: int, out_w: int):
        """
        采样 (i,j)，使裁剪窗四边全为背景（黑=0），避免窗内任何前景连通域被截断；
        若 ensure_fg=True，还要求窗内至少含有一个前景像素。
        """
        # 将三通道 mask 二值化：>0.5 视为白(前景)=1
        mb = (M.mean(dim=0) > 0.5).to(torch.uint8)  # [H,W] 1=前景, 0=背景(黑)

        H, W = mb.shape
        max_i, max_j = H - out_h, W - out_w
        if max_i < 0 or max_j < 0:
            return 0, 0  # 兜底

        for _ in range(self.component_retries):
            i = random.randint(0, max_i)
            j = random.randint(0, max_j)
            crop = mb[i:i+out_h, j:j+out_w]

            # 四边必须为背景（0），否则可能截断前景连通域
            if crop[0, :].any():     continue
            if crop[-1, :].any():    continue
            if crop[:, 0].any():     continue
            if crop[:, -1].any():    continue

            if self.ensure_fg and crop.sum().item() == 0:
                continue  # 要求窗内必须有前景

            return i, j

        # 回退：普通随机裁剪
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
        return i, j



@Registers.datasets.register_with_name('change_detection_layout')
class ChangeDetectionLayoutDataset(Dataset):
    """
    等价于原 ChangeDetectionDataset 中 condition_type == "instance layout" 的行为
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.to_normal = dataset_config.to_normal

        # 定义路径
        list_dir = os.path.join(dataset_config.dataset_path, 'list')
        A_dir = os.path.join(dataset_config.dataset_path, 'A')
        B_dir = os.path.join(dataset_config.dataset_path, 'B')
        mask_dir = os.path.join(dataset_config.dataset_path, 'label')

        # 获取图像对
        self.image_pairs = get_image_triplets_from_list(list_dir, stage, A_dir, B_dir, mask_dir)
        self.original_length = len(self.image_pairs)
        self._length = self.original_length * 2  # 数据集长度翻倍

        # 初始化同步变换
        self.transform_original = SyncTransform(
            self.image_size, stage,
            scale_range=(1.0, 1.2),  # 原样本：轻微放大
            component_safe=False,  # 如需完整性可改 True
            ensure_fg=False
        )
        self.transform_scaled = SyncTransform(
            self.image_size, stage,
            scale_range=(1.5, 2.5),  # 增强样本：更大随机多尺度
            component_safe=False,
            ensure_fg=False
        )
        # 兼容老写法（固定 2x）： self.transform_scaled = SyncTransform(self.image_size, stage, is_scaled=True)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # 确定是原始样本还是增强样本
        is_scaled = index >= self.original_length
        real_index = index - self.original_length if is_scaled else index

        A_path, B_path, mask_path = self.image_pairs[real_index]

        # 加载图像 A、B 和 mask
        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        mask_ori = Image.open(mask_path).convert('RGB')

        # 应用同步变换
        transform = self.transform_scaled if is_scaled else self.transform_original
        image_A, image_B, mask_ori = transform(image_A, image_B, mask_ori)

        # 归一化图像和掩码（在所有阶段如果 to_normal 为 True）
        if self.to_normal:
            image_A = (image_A - 0.5) * 2.0
            image_A = image_A.clamp(-1.0, 1.0)
            image_B = (image_B - 0.5) * 2.0
            image_B = image_B.clamp(-1.0, 1.0)
            mask_ori = (mask_ori - 0.5) * 2.0  # 归一化掩码
            mask_ori = mask_ori.clamp(-1.0, 1.0)

        # === 布局逻辑（原 condition_type == "instance layout"）===
        mask_swapped = mask_ori.clone()
        # 仅匹配黑色区域（-1.0），把这些区域替换为 image_A 的内容
        black_mask = torch.isclose(mask_ori, torch.tensor(-1.0, device=mask_ori.device, dtype=mask_ori.dtype))
        mask_swapped = torch.where(black_mask, image_A, mask_ori)

        # 提取图像名称
        image_name = Path(A_path).stem

        # 调试打印（保持原始行为）
        print("image_B:", image_B)
        print("image_A:", image_A)
        print("mask_swapped:", mask_swapped)
        print("mask_ori:", mask_ori)

        # 返回结构保持不变
        # return (image_B, image_name), (mask_swapped, image_name), (image_A, image_name), (mask_ori, image_name)
        return (image_B, image_name), (image_A, image_name), (mask_swapped, image_name), (mask_ori, image_name)
        # (x, x_name), (x_cond, x_cond_name), (context, context_name), (mask, image_name)


@Registers.datasets.register_with_name('change_detection_layout')
class ChangeDetectionLayoutDataset(Dataset):
    """
    布局版（对应原 condition_type == "instance layout"）
    - 不将 original/scaled 成对放同批：保持长度翻倍，但用普通 DataLoader 随机采样即可
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.to_normal = dataset_config.to_normal

        # 路径与索引
        list_dir = os.path.join(dataset_config.dataset_path, 'list')
        A_dir    = os.path.join(dataset_config.dataset_path, 'A')
        B_dir    = os.path.join(dataset_config.dataset_path, 'B')
        mask_dir = os.path.join(dataset_config.dataset_path, 'label')
        self.image_pairs = get_image_triplets_from_list(list_dir, stage, A_dir, B_dir, mask_dir)

        self.original_length = len(self.image_pairs)
        self._length = self.original_length * 2  # 数据集长度翻倍（前半 original，后半 scaled）

        # 同步增强（纯几何；大视野随机多尺度更强；不强制 component-safe）
        self.transform_original = SyncTransform(
            self.image_size, stage,
            scale_range=(1.0, 1.2),
            component_safe=False,   # 如需“连通域完整”可改 True
            ensure_fg=False
        )
        self.transform_scaled = SyncTransform(
            self.image_size, stage,
            scale_range=(1.5, 2.5),
            component_safe=False,
            ensure_fg=False
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # 前半段用 original 增强；后半段用 scaled 增强
        is_scaled = index >= self.original_length
        real_index = index - self.original_length if is_scaled else index

        A_path, B_path, mask_path = self.image_pairs[real_index]

        # 读取
        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        mask_ori = Image.open(mask_path).convert('RGB')

        # 同步变换（输出均为 [0,1]）
        transform = self.transform_scaled if is_scaled else self.transform_original
        image_A, image_B, mask_ori = transform(image_A, image_B, mask_ori)

        # —— 使用布尔 mask（更快更稳）——
        # m_bin: [1,H,W]，True 表示白(前景=1)，False 表示黑(背景=0)
        m_bin = (mask_ori.mean(dim=0, keepdim=True) > 0.5)
        m3 = m_bin.expand_as(image_A)  # [3,H,W] 仅创建视图，无额外内存

        # 归一化到 [-1,1]（如配置需要）
        if self.to_normal:
            image_A = (image_A - 0.5) * 2.0; image_A.clamp_(-1.0, 1.0)
            image_B = (image_B - 0.5) * 2.0; image_B.clamp_(-1.0, 1.0)
            mask_img = m_bin.float().expand_as(image_A)  # 0/1
            mask_ori = mask_img * 2.0 - 1.0              # 黑=-1 白=+1
        else:
            mask_ori = m_bin.float().expand_as(image_A)  # 保持 0/1

        # 布局逻辑：仅“黑区”（背景）用 A 的内容替换
        mask_swapped = torch.where(~m3, image_A, mask_ori)

        image_name = Path(A_path).stem

        # return (image_B, image_name), (mask_swapped, image_name), (image_A, image_name), (mask_ori, image_name)
        return (image_B, image_name), (image_A, image_name), (mask_swapped, image_name), (mask_ori, image_name)
        # (x, x_name), (x_cond, x_cond_name), (context, context_name), (mask, image_name)



@Registers.datasets.register_with_name('change_detection_semantic')
class ChangeDetectionSemanticDataset(Dataset):
    """
    语义版（对应原 condition_type == "semantic map"）
    - 不将 original/scaled 成对放同批：保持长度翻倍，但用普通 DataLoader 随机采样即可
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.to_normal = dataset_config.to_normal

        list_dir = os.path.join(dataset_config.dataset_path, 'list')
        A_dir    = os.path.join(dataset_config.dataset_path, 'A')
        B_dir    = os.path.join(dataset_config.dataset_path, 'B')
        mask_dir = os.path.join(dataset_config.dataset_path, 'label')
        self.image_pairs = get_image_triplets_from_list(list_dir, stage, A_dir, B_dir, mask_dir)

        self.original_length = len(self.image_pairs)
        self._length = self.original_length * 2

        # 语义更看细节：多尺度轻一些，且建议 component_safe=True（如你需要完整性）
        self.transform_original = SyncTransform(
            self.image_size, stage,
            scale_range=(1.0, 1.0),
            component_safe=True,
            ensure_fg=True,
            component_retries=20
        )
        self.transform_scaled = SyncTransform(
            self.image_size, stage,
            scale_range=(1.0, 1.15),
            component_safe=True,
            ensure_fg=True,
            component_retries=20
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        is_scaled = index >= self.original_length
        real_index = index - self.original_length if is_scaled else index

        A_path, B_path, mask_path = self.image_pairs[real_index]
        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        mask_ori = Image.open(mask_path).convert('RGB')

        transform = self.transform_scaled if is_scaled else self.transform_original
        image_A, image_B, mask_ori = transform(image_A, image_B, mask_ori)

        # 布尔 mask & 归一化
        m_bin = (mask_ori.mean(dim=0, keepdim=True) > 0.5)
        m3 = m_bin.expand_as(image_A)

        if self.to_normal:
            image_A = (image_A - 0.5) * 2.0; image_A.clamp_(-1.0, 1.0)
            image_B = (image_B - 0.5) * 2.0; image_B.clamp_(-1.0, 1.0)
            mask_img = m_bin.float().expand_as(image_A)
            mask_ori = mask_img * 2.0 - 1.0
        else:
            mask_ori = m_bin.float().expand_as(image_A)

        # 语义逻辑：仅“白区”（前景）用 A 的内容替换
        mask_swapped = torch.where(m3, image_A, mask_ori)

        image_name = Path(A_path).stem

        # return (image_B, image_name), (mask_swapped, image_name), (image_A, image_name), (mask_ori, image_name)
        return (image_B, image_name), (image_A, image_name), (mask_swapped, image_name), (mask_ori, image_name)
        # (x, x_name), (x_cond, x_cond_name), (context, context_name), (mask, image_name)


# 数据集定义
@Registers.datasets.register_with_name('change_detection')
class ChangeDetectionDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.to_normal = dataset_config.to_normal

        # 确保 dataset_config 中必须提供 condition_type，否则报错
        if not hasattr(dataset_config, "condition_type"):
            raise ValueError("dataset_config 必须包含 'condition_type'")

        self.condition_type = dataset_config.condition_type  # 直接从 dataset_config 读取

        # 定义路径
        list_dir = os.path.join(dataset_config.dataset_path, 'list')
        A_dir = os.path.join(dataset_config.dataset_path, 'A')
        B_dir = os.path.join(dataset_config.dataset_path, 'B')
        mask_dir = os.path.join(dataset_config.dataset_path, 'label')

        # 获取图像对
        self.image_pairs = get_image_triplets_from_list(list_dir, stage, A_dir, B_dir, mask_dir)
        self.original_length = len(self.image_pairs)
        self._length = self.original_length * 2  # 数据集长度翻倍

        # 初始化同步变换
        self.transform_original = SyncTransform(self.image_size, stage)
        self.transform_scaled = SyncTransform(self.image_size, stage, is_scaled=True)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # 确定是原始样本还是增强样本
        is_scaled = index >= self.original_length
        real_index = index - self.original_length if is_scaled else index

        A_path, B_path, mask_path = self.image_pairs[real_index]

        # 加载图像 A、B 和 mask
        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        mask_ori = Image.open(mask_path).convert('RGB')


        # 应用同步变换
        transform = self.transform_scaled if is_scaled else self.transform_original
        image_A, image_B, mask_ori = transform(image_A, image_B, mask_ori)


        # 归一化图像和掩码（在所有阶段如果 to_normal 为 True）
        ### for semantic labels
        if self.to_normal:
            image_A = (image_A - 0.5) * 2.0
            image_A = image_A.clamp(-1.0, 1.0)
            image_B = (image_B - 0.5) * 2.0
            image_B = image_B.clamp(-1.0, 1.0)
            mask_ori = (mask_ori - 0.5) * 2.0  # 归一化掩码
            mask_ori = mask_ori.clamp(-1.0, 1.0)

###
        # 创建 mask_swapped
        mask_swapped = mask_ori.clone()

        if self.condition_type == "semantic map":
            # 仅匹配白色区域（1.0）
            white_mask = torch.isclose(mask_ori, torch.tensor(1.0, device=mask_ori.device, dtype=mask_ori.dtype))
            # 仅改变白色背景区域的内容，其他区域保持不变
            # 将白色区域替换为 image_A 的内容，其他区域保留原始图像的内容（mask_ori）
            mask_swapped = torch.where(white_mask, image_A, mask_ori)
            # 将掩码中的白色区域（1.0）替换为黑色（-1.0）
            # mask_ori[torch.isclose(mask_ori, torch.tensor(1.0, device=mask_ori.device, dtype=mask_ori.dtype))] = -1.0
            # # print('mask_ori:', mask_ori)

        elif self.condition_type == "instance layout":
            # 仅匹配黑色区域（-1.0）
            black_mask = torch.isclose(mask_ori, torch.tensor(-1.0, device=mask_ori.device, dtype=mask_ori.dtype))
            # 将黑色区域替换为 image_A
            mask_swapped = torch.where(black_mask, image_A, mask_ori)
            # print('mask_ori:', mask_ori)

        else:
            raise ValueError(f"未知的 condition_type: {self.condition_type}")
###

        # 提取图像名称
        image_name = Path(A_path).stem

        # 打印这四个变量的值
        print("image_B:", image_B)
        print("image_A:", image_A)
        print("mask_swapped:", mask_swapped)
        print("mask_ori:", mask_ori)


        # 返回数据样本
        # return (image_B, image_name), (mask_swapped, image_name), (image_A, image_name), (mask_ori, image_name)
        return (image_B, image_name), (image_A, image_name), (mask_swapped, image_name), (mask_ori, image_name)
        # (x, x_name), (x_cond, x_cond_name), (context, context_name), (mask, image_name)
###




