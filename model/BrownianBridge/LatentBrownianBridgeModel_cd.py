import itertools
import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel_cd import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")
###
        # 初始化1x1卷积层，不使用激活函数
        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(
        #     in_channels=model_config.in_channels * 2,
        #     out_channels=model_config.in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False  # 禁用偏置
        # ),
        #     nn.SiLU()  # 非线性激活函数
        # )

        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=model_config.in_channels,
        #     num_heads=3,
        #     batch_first=True
        # )

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

###
    # def cross_attention(self, query, key, value):
    #     """
    #     封装跨注意力机制，执行维度转换和跨注意力计算。
    #     """
    #     # 将输入转换为 [batch_size, seq_len, embed_dim] 格式
    #     query = query.flatten(2).permute(0, 2, 1)
    #     key = key.flatten(2).permute(0, 2, 1)
    #     value = value.flatten(2).permute(0, 2, 1)
    #
    #     # 执行跨注意力计算
    #     attn_out, _ = self.cross_attn(query, key, value)
    #
    #     # 转回原始形状
    #     attn_out = attn_out.permute(0, 2, 1).view_as(query.permute(0, 2, 1))
    #
    #     return attn_out

###


    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x2, x1, control=None, mask=None):
        with torch.no_grad():
            x_latent = self.encode(x2, cond=False)
            cond_latent = self.encode(x1, cond=True)

            # x1 = self.encode(x1)
####
        # 使用 T1 的潜在特征作为 Key 和 Value，差值作为 Query 进行跨注意力, 生成的潜在特征
        # x_latent = self.cross_attention(query=x_latent, key=cond_latent, value=cond_latent)
####
            print(f"context type: {type(control)}, shape: {control.shape}")

            control = self.get_cond_stage_context(control)

###
            fg_mask = self.get_cond_stage_context(control)
            bg_mask = self.get_cond_stage_context(control)
###

        return super().forward(x_latent.detach(), cond_latent.detach(), control=control, mask=mask)

    def get_cond_stage_context(self, context):  # 输入 context或者x1_cond
        if self.cond_stage_model is not None:
            # "nocond"时候，执行这句，输入torch.cat([image_A, mask]
            context = self.cond_stage_model(context)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

###
    @torch.no_grad()
    def sample(self, cond, x1, mask, clip_denoised=False, sample_mid_step=False):
        """
        推理阶段：使用 T1 和 change mask 生成 T2。
        """

        # 编码 T1 和 change mask
        t1_latent = self.get_cond_stage_context(x1)  # 使用 context
        mask_latent = self.encode(cond, cond=True)

        # 使用 p_sample_loop 生成潜在特征 (difference latent)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=mask_latent, context=t1_latent, mask=mask, clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)

            # 解码生成的中间步骤潜在特征
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    # attn_out = self.cross_attention(temp[i].detach(), t1_latent, t1_latent)  # 使用封装的跨注意力函数
                    #t2_generated = self.decode(attn_out, cond=False)
                    t2_generated = self.decode(temp[i].detach(), cond=False)
###
                out_samples.append(t2_generated.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True, smoothing=0.01):
                with torch.no_grad():
                    # attn_out = self.cross_attention(one_step_temp[i].detach(), t1_latent, t1_latent)  # 使用封装的跨注意力函数
                    # t2_generated = self.decode(attn_out, cond=False)
                    t2_generated = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(t2_generated.to('cpu'))

            return out_samples, one_step_samples

        else:
            # 如果不需要中间步骤，则直接解码生成最终的 T2
            generated_latent = self.p_sample_loop(y=mask_latent, context=t1_latent, mask=mask, clip_denoised=clip_denoised)

            # 使用封装的跨注意力函数
# ###
#             print("generated_latent:", generated_latent)
#             print("t1_latent:", t1_latent)
# ###

            # generated_latent = self.cross_attention(generated_latent, t1_latent, t1_latent)

            # 解码生成 T2 图像
            t2_generated = self.decode(generated_latent, cond=False)

            return t2_generated

    # @torch.no_grad()
    # def sample_intermediate(self, cond, x1, clip_denoised=False, steps_to_save=[20, 40, 60, 80, 100]):
    #     """
    #     执行 Diffusion 过程并在每隔 `steps_to_save` 的位置保存中间结果。
    #     """
    #     # 编码 T1 和 change mask
    #     t1_latent = self.get_cond_stage_context(x1)  # 使用 context
    #     mask_latent = self.encode(cond, cond=True)
    #
    #     # 使用 p_sample_loop 生成潜在特征 (difference latent)
    #     temp, one_step_temp = self.p_sample_loop(y=mask_latent,
    #                                              context=t1_latent,
    #                                              clip_denoised=clip_denoised,
    #                                              sample_mid_step=True)
    #
    #     # 保存每个步骤的生成图像
    #     out_samples = []
    #     one_step_samples = []
    #
    #     for step in range(len(temp)):
    #         if step in steps_to_save:
    #             # 解码生成的中间步骤潜在特征
    #             for i in range(len(temp)):
    #                 with torch.no_grad():
    #                     t2_generated = self.decode(temp[i].detach(), cond=False)
    #                     out_samples.append(t2_generated.to('cpu'))
    #
    #         if step in steps_to_save:
    #             # 对于 one step 的图像，也做相同处理
    #             for i in range(len(one_step_temp)):
    #                 with torch.no_grad():
    #                     t2_generated = self.decode(one_step_temp[i].detach(), cond=False)
    #                     one_step_samples.append(t2_generated.to('cpu'))
    #
    #     return out_samples, one_step_samples

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
