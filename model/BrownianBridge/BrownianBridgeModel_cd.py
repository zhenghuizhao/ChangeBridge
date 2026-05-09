import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler


class BrownianBridgeModel(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.eta_fg = model_params.eta_fg if model_params.__contains__("eta_fg") else self.eta  # 前景的 eta
        self.eta_bg = model_params.eta_bg if model_params.__contains__("eta_bg") else self.eta * 0.6  # 背景的 eta
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps
        m_min, m_max = 0.001, 0.999  # 统一漂移的最大值（无前景和背景的区别）

        # 构造统一的 m_t（线性或正弦）
        if self.mt_type == "linear":
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = m_max
        else:
            raise NotImplementedError

        # m_tminus（前一时刻）
        m_tminus = np.append(0, m_t[:-1])

        # Step 1: 统一 variance 和 beta
        beta = 2. * (m_t - m_t ** 2) * self.max_var
        var = beta.copy()

        # t-1 的 variance
        var_tminus = np.append(0., var[:-1])

        # Posterior variance 推导（Brownian Bridge 近似）
        var_t_tminus = var - var_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_var = var_t_tminus * var_tminus / var

        # 注册到模型
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 统一的 m_t 和 variance
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(var))
        self.register_buffer('variance_tminus', to_torch(var_tminus))
        self.register_buffer('variance_t_tminus', to_torch(var_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_var))

        # 采样步数（保留）
        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x2, x1, control=None, mask=None):
        b, c, h, w, device, img_size, = *x2.shape, x2.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x2, x1, control, t, mask)

    def p_losses(self, x0, y, context, t, mask, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param context: conditioning context (e.g., mask or other guidance)
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss, fg_loss, bg_loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        # 传递 context 给 q_sample
        x_t, objective = self.q_sample(x0, y, t, mask=mask, noise=noise)

        # 反向预测
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=y)

        # 计算重建损失
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        # 重建 x0（可选）
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

        # 使用mask计算前景和背景的损失
        fg_mask = (mask != 1.).float()  # 前景mask
        bg_mask = (mask == 1.).float()  # 背景mask

        # 计算前景损失
        fg_loss = (fg_mask * (objective - objective_recon).abs()).mean()

        # 计算背景损失
        bg_loss = (bg_mask * (objective - objective_recon).abs()).mean()

        print(f"Foreground Loss: {fg_loss.item()}")  # 打印前景损失
        print(f"Background Loss: {bg_loss.item()}")  # 打印背景损失

        # 记录日志
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon,
            "fg_loss": fg_loss,
            "bg_loss": bg_loss
        }

        # 返回总损失，前景损失和背景损失
        return recloss, log_dict

    def q_sample(self, x0, y, t, mask, noise=None):
        """
        统一漂移：前景和背景使用相同的 drift 和 variance
        并分别计算噪声（噪声不同步）
        """
        assert mask is not None, "上下文 mask 不能为空，用于掩码判断"
        noise = default(noise, lambda: torch.randn_like(x0))

        # 为前景和背景区域设置不同的 eta
        fg_eta = self.eta  # 前景的 eta
        bg_eta = self.eta * 1.0  # 背景的 eta，设置为前景的 60%

        fg_mask = (mask != 1.).float()  # 前景
        bg_mask = (mask == 1.).float()  # 背景

        # 提取统一的 m_t 和 sigma_t
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        # 根据前景和背景的噪声强度，分别调整噪声幅度
        fg_noise = fg_mask * (noise * fg_eta)
        bg_noise = bg_mask * (noise * bg_eta)

        # 对噪声进行归一化（确保噪声有零均值和单位方差）
        # fg_noise = (fg_noise - fg_noise.mean()) / (fg_noise.std() + 1e-8)
        # bg_noise = (bg_noise - bg_noise.mean()) / (bg_noise.std() + 1e-8)

        # 构造目标值
        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * (fg_noise + bg_noise)
        elif self.objective == 'noise':
            objective = fg_noise + bg_noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        # 构造 q(x_t | x0, y)
        x_t = (1. - m_t) * x0 + m_t * y + sigma_t * (fg_noise + bg_noise)
        return x_t, objective

    def predict_x0_from_objective(self, x_t, cond, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * cond - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = cond - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, mask, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device

        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=y)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            # 当前时间步与下一个时间步的时间戳
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i + 1], device=x_t.device, dtype=torch.long)

            # 使用去噪模型对当前噪声图像进行去噪
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=y)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            # 获取时间步 t 和 n_t 的相关参数
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)

            # 计算方差的平方（标准差的平方）
            sigma2_t = var_t  # 使用当前时间步的方差 sigma^2
            sigma_t = torch.sqrt(sigma2_t)  # 标准差

            # 前景和背景的掩码
            fg_mask = (mask != 1.).float()  # 前景
            bg_mask = (mask == 1.).float()  # 背景

            # 从加噪过程（q_sample）中获取噪声生成的方式
            fg_noise = fg_mask * torch.randn_like(x_t)  # 前景噪声
            bg_noise = bg_mask * torch.randn_like(x_t)  # 背景噪声

            # 为前景和背景分别设置不同的噪声强度（前景和背景的 eta 值不同）
            fg_eta = self.eta  # 前景的 eta
            bg_eta = self.eta * 0.6  # 背景的 eta（设置为前景的 60%）

            # 根据噪声强度调整噪声（反向过程的噪声去除）
            fg_denoised = fg_noise * sigma_t * fg_eta
            bg_denoised = bg_noise * sigma_t * bg_eta

            # 使用时间步和噪声强度恢复 x_t 到 x_{t-1}
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            # 合并前景和背景的去噪结果
            return x_tminus_mean + fg_denoised + bg_denoised, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, mask=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, mask=mask, i=i,
                                              clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, mask=mask, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, mask=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, mask, clip_denoised=clip_denoised, sample_mid_step=sample_mid_step)
