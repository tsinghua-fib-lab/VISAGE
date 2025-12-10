import torch.nn as nn
import re
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
import torch


import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        初始化 Multi-Head Attention
        :param dim: 输入向量的维度
        :param heads: 多头数量
        :param dim_head: 每个头的维度（如果不指定，默认为 dim // heads）
        """
        super(MultiHeadAttention, self).__init__()
        
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else dim // heads
        inner_dim = self.dim_head * heads
        self.scale = self.dim_head ** -0.5  # 缩放因子

        # LayerNorm 用于规范输入的 query 和 key-value
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        # 线性变换：将输入映射到 query 和 key-value 的内部维度
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # 用于生成 query
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # 用于生成 key 和 value

        # 输出层
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)            
        self.apply(_basic_init)  
        
        
    def forward(self, q, kv):
        """
        计算 Multi-Head Attention
        :param si_features: [bs, hiddenstate_size] 作为 query
        :param sv_features: [bs, hiddenstate_size] 作为 key 和 value
        :return: context 向量 [bs, hiddenstate_size]
        """
        bs ,patch_size, hidden_size = kv.size()
        
        

        
        
        q = self.norm_media(q)
        kv = self.norm_latents(kv)

        # 生成 query, key, value
        query = self.to_q(q)  # [bs, inner_dim]
        kv = self.to_kv(kv)  # [bs, inner_dim * 2]
        key, value = kv.chunk(2, dim=-1)  # 拆分成 key 和 value [bs, inner_dim]

        # 调整 query, key, value 的形状
        query = query.view(bs, patch_size,self.heads, self.dim_head)#.transpose(1,2)
        key = key.view(bs, patch_size,self.heads, self.dim_head)#.transpose(1,2) 
        value = value.view(bs,patch_size, self.heads, self.dim_head)#.transpose(1,2)  


        # 计算注意力分数，并应用缩放
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # [bs, heads, heads]
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # [bs, heads, heads]

        # 应用注意力权重到 value
        context = torch.matmul(attention_weights, value)  # [bs, 729,heads, dim_head]
        # 将多头拼接
        context = context.contiguous().view(bs,patch_size,self.heads * self.dim_head)  # [bs, inner_dim]

        # 最终的线性输出
        output = self.to_out(context)  # [bs, dim]
        return output




class PerceiverResampler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        #self.pretrained = model_args.mm_perceiver_pretrained

        hidden_size = hidden_size
        self.si2sv = MultiHeadAttention(dim=hidden_size, heads=8)
        self.sv2si = MultiHeadAttention(dim=hidden_size, heads=8)
        
        
        self.sigate = nn.Linear(hidden_size,hidden_size, bias=True)
        self.svgate = nn.Linear(hidden_size, hidden_size, bias=True)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)            
        self.apply(_basic_init)  
        nn.init.constant_(self.sigate.weight,0)
        nn.init.constant_(self.sigate.bias,0)
        nn.init.constant_(self.svgate.weight,0)
        nn.init.constant_(self.svgate.bias,0)
        
    def forward(self, image_features, images_num, *args, **kwargs):
        B, patch_size, hidden_size = image_features.size()
        si_idx = 0
        si_features = []
        sv_features = []
        
        for i, nums in enumerate(images_num):
            # 处理单张图像的情况
            si_features.append(image_features[si_idx:si_idx+1, :, :])
            
            if nums != 1:
                # 处理多张图像的情况
                sv_features.append(torch.mean(image_features[si_idx+1:si_idx + nums, :, :], dim=0, keepdim=True))
            else:
                # 对于单张图像，不使用零填充，而是直接保留其自身特征
                sv_features.append(image_features[si_idx:si_idx+1, :, :])
            
            si_idx = si_idx + nums
        
        # 将收集的特征拼接
        si_features = torch.cat(si_features, dim=0)
        sv_features = torch.cat(sv_features, dim=0)
        
        # 使用多头注意力机制进行特征交互
        sv_context = self.si2sv(si_features, sv_features)
        si_context = self.sv2si(sv_features, si_features)
        # 更新 image_features，避免就地操作
        updated_features = image_features.clone()
        si_idx = 0
        for i, nums in enumerate(images_num):
            if nums != 1:
                # 更新主图像特征
                updated_features[si_idx:si_idx+1] = updated_features[si_idx:si_idx+1] + self.sigate(si_context[i:i+1])
                # 更新辅助图像特征
                updated_features[si_idx+1:si_idx + nums] = updated_features[si_idx+1:si_idx + nums] + self.svgate(sv_context[i:i+1]).repeat(nums-1, 1, 1)
            else:
                # 更新单张图像特征
                updated_features[si_idx:si_idx+1] = updated_features[si_idx:si_idx+1] + (self.sigate(si_context[i:i+1])+self.svgate(sv_context[i:i+1]))/2
            
            si_idx += nums
        
        return updated_features


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class DownSampleBlock(nn.Module):

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x

class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str=None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type
        self.perceiver =None


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig
    def __init__(
        self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig
    ):
        super().__init__(mm_projector_cfg)
        self.perceiver = None
        mm_projector_type = mm_projector_cfg.mm_projector_type
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif mm_projector_type == "mlp_downsample":
            self.perceiver=PerceiverResampler(config.mm_hidden_size)
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x,image_nums, *args, **kwargs):
        x = self.perceiver(x,image_nums)

        
        return self.layers(x)

AutoConfig.register("v2l_projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)