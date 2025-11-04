"""
OPT模型的INT8量化实现模块

本模块实现了OPT (Open Pre-trained Transformer) 模型的INT8量化版本，用于SmoothQuant量化方案。
主要功能是将FP32浮点模型转换为INT8量化模型，以减少内存占用并加速推理。

主要组件：
- Int8OPTAttention: INT8量化的多头注意力层
- Int8OPTDecoderLayer: INT8量化的解码器层
- Int8OPTDecoder: INT8量化的解码器
- Int8OPTModel: INT8量化的模型
- Int8OPTForCausalLM: INT8量化的因果语言模型
"""
import torch
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTAttention,
    OPTDecoderLayer,
    OPTDecoder,
    BaseModelOutputWithPast,
)
from typing import Optional, Tuple, List
# INT8量化线性层：
# W8A8B8O8Linear: 权重(W)和激活(A)都是INT8，偏置(B)和输出(O)也是INT8
# W8A8BFP32OFP32Linear: 权重和激活是INT8，偏置和输出是FP32
# W8A8B8O8LinearReLU: 带ReLU激活的INT8线性层
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ  # 量化版本的LayerNorm
from transformers.utils import logging
# INT8矩阵乘法：
# BMM_S8T_S8N_S8T: INT8转置 x INT8非转置 -> INT8转置
# BMM_S8T_S8N_F32T: INT8转置 x INT8非转置 -> FP32转置
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T

logger = logging.get_logger(__name__)


class Int8OPTAttention(nn.Module):
    """
    INT8量化的多头注意力层
    
    实现了OPT模型的多头注意力机制的INT8量化版本。
    使用INT8量化来减少计算和内存开销，同时保持较高的精度。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        """
        初始化INT8注意力层
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 验证embed_dim能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        # INT8矩阵乘法：Query-Key矩阵乘法，输出FP32
        # 用于计算注意力权重 (Q @ K^T)
        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        # INT8矩阵乘法：注意力概率-Value矩阵乘法，输出INT8
        # 用于计算注意力输出 (Attn_Probs @ V)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        # INT8量化的投影层：权重和激活都是INT8
        self.k_proj = W8A8B8O8Linear(embed_dim, embed_dim)  # Key投影
        self.v_proj = W8A8B8O8Linear(embed_dim, embed_dim)  # Value投影
        self.q_proj = W8A8B8O8Linear(embed_dim, embed_dim)  # Query投影
        # 输出投影：输入INT8，输出FP32（因为需要与残差连接）
        self.out_proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)

    @staticmethod
    @torch.no_grad()
    def from_float(
        module: OPTAttention,
        input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
    ):
        """
        从FP32浮点注意力模块转换为INT8量化模块
        
        Args:
            module: FP32的OPTAttention模块
            input_scale: 输入张量的量化缩放因子
            q_output_scale: Query投影输出的量化缩放因子
            k_output_scale: Key投影输出的量化缩放因子
            v_output_scale: Value投影输出的量化缩放因子
            out_input_scale: 输出投影输入的量化缩放因子
            
        Returns:
            INT8量化的注意力模块
        """
        int8_module = Int8OPTAttention(module.embed_dim, module.num_heads)
        
        # 将OPT的scaling因子融合到q_proj的输出缩放中
        # 这样可以避免在运行时进行额外的缩放操作
        q_output_scale = q_output_scale * module.scaling
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling
        
        # 将FP32投影层转换为INT8量化层
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale
        )
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale
        )
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale
        )
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale
        )
        
        # 设置Q@K^T矩阵乘法的缩放因子
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(q_output_scale, k_output_scale)

        # 设置注意力概率@Value矩阵乘法的缩放因子
        # alpha = s_prob * s_v / s_out
        # 其中 s_prob = 1 / 127 (softmax后的概率量化到[-127, 127])
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale
        )
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        重塑张量形状以适配多头注意力计算
        
        将形状从 (bsz, seq_len, embed_dim) 转换为 (bsz, num_heads, seq_len, head_dim)
        
        Args:
            tensor: 输入张量
            seq_len: 序列长度
            bsz: 批次大小
            
        Returns:
            重塑后的张量，形状为 (bsz, num_heads, seq_len, head_dim)
        """
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        输入形状: Batch x Time x Channel
        
        Args:
            hidden_states: 输入的隐藏状态，形状为 (bsz, tgt_len, embed_dim)
            key_value_states: 用于交叉注意力的键值状态（可选）
            past_key_value: 缓存的过去键值对，用于加速解码（可选）
            attention_mask: 注意力掩码，用于屏蔽padding位置（可选）
            layer_head_mask: 层级的注意力头掩码（可选）
            output_attentions: 是否返回注意力权重
            
        Returns:
            attn_output: 注意力输出
            attn_probs_reshaped: 注意力概率（如果output_attentions=True）
            past_key_value: 更新后的键值对缓存
        """
        # 判断是否为交叉注意力（用于decoder）
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # 计算Query投影（INT8量化）
        query_states = self.q_proj(hidden_states)
        
        # 计算Key和Value投影（INT8量化）
        # 处理不同的注意力模式：交叉注意力、缓存的自注意力、普通自注意力
        if is_cross_attention and past_key_value is not None:
            # 交叉注意力且使用缓存的k,v
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # 交叉注意力，从key_value_states计算k,v
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # 自注意力且使用缓存：将新的k,v与缓存的k,v拼接
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # 普通自注意力
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        # 重塑为多头注意力格式: (bsz * num_heads, seq_len, head_dim)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # 计算注意力权重: Q @ K^T (INT8矩阵乘法，输出FP32)
        src_len = key_states.size(1)
        attn_weights = self.qk_bmm(query_states, key_states)

        # 验证注意力权重的形状
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # 将掩码加到注意力权重上（padding位置会变成很大的负值）
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            # 将padding位置的权重设为最小值（softmax后会接近0）
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # 计算注意力概率（softmax）
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        # 应用层级注意力头掩码（如果有）
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        # 如果需要返回注意力权重，进行重塑（保持梯度）
        if output_attentions:
            # 这个操作有点繁琐，但为了保持attn_weights的梯度是必需的
            # 需要重塑两次并在后续复用
            attn_probs_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_probs_reshaped = None

        # 将注意力概率量化为INT8
        # 乘以127并四舍五入，将[0, 1]的概率值映射到[0, 127]的INT8范围
        # 公式: (A_row V_row)_row = (A_row V_col^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        # 计算注意力输出: Attn_Probs @ V (INT8矩阵乘法)
        value_states = value_states.transpose(1, 2).contiguous()
        attn_output = self.pv_bmm(attn_probs, value_states)

        # 验证注意力输出的形状
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 重塑并转置回原始形状: (bsz, tgt_len, num_heads, head_dim) -> (bsz, tgt_len, num_heads, head_dim)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # 重塑为最终输出形状: (bsz, tgt_len, embed_dim)
        # 使用类中存储的embed_dim而不是hidden_state的维度，因为在使用张量并行时
        # attn_output可能被分割到多个GPU上
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim).contiguous()
        
        # 通过输出投影层（INT8输入，FP32输出，用于残差连接）
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value


class Int8OPTDecoderLayer(nn.Module):
    """
    INT8量化的OPT解码器层
    
    包含：
    - 自注意力层（INT8量化）
    - 前馈网络（FFN，INT8量化）
    - LayerNorm层（量化版本）
    """
    def __init__(self, embed_dim, num_attention_heads, ffn_dim):
        """
        初始化INT8解码器层
        
        Args:
            embed_dim: 嵌入维度
            num_attention_heads: 注意力头数
            ffn_dim: 前馈网络的隐藏层维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # INT8量化的自注意力层
        self.self_attn = Int8OPTAttention(
            embed_dim=self.embed_dim, num_heads=num_attention_heads
        )

        # 量化版本的LayerNorm（注意力层前）
        self.self_attn_layer_norm = LayerNormQ(self.embed_dim)
        
        # FFN的第一层：INT8输入输出，带ReLU激活
        self.fc1 = W8A8B8O8LinearReLU(self.embed_dim, ffn_dim)
        
        # FFN的第二层：INT8输入，FP32输出（用于残差连接）
        self.fc2 = W8A8BFP32OFP32Linear(ffn_dim, self.embed_dim)
        
        # 量化版本的LayerNorm（FFN前）
        self.final_layer_norm = LayerNormQ(self.embed_dim)

    @staticmethod
    def from_float(
        module: OPTDecoderLayer,
        attn_input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
        fc1_input_scale: float,
        fc2_input_scale: float,
    ):
        """
        从FP32浮点解码器层转换为INT8量化层
        
        Args:
            module: FP32的OPTDecoderLayer模块
            attn_input_scale: 注意力层输入的量化缩放因子
            q_output_scale: Query投影输出的量化缩放因子
            k_output_scale: Key投影输出的量化缩放因子
            v_output_scale: Value投影输出的量化缩放因子
            out_input_scale: 注意力输出投影输入的量化缩放因子
            fc1_input_scale: FFN第一层输入的量化缩放因子
            fc2_input_scale: FFN第二层输入的量化缩放因子
            
        Returns:
            INT8量化的解码器层
        """
        int8_module = Int8OPTDecoderLayer(
            module.embed_dim, module.self_attn.num_heads, module.fc1.out_features
        )
        
        # 转换LayerNorm层（注意力层前）
        int8_module.self_attn_layer_norm = LayerNormQ.from_float(
            module.self_attn_layer_norm, attn_input_scale
        )
        
        # 转换注意力层
        int8_module.self_attn = Int8OPTAttention.from_float(
            module.self_attn,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale,
        )
        
        # 转换LayerNorm层（FFN前）
        int8_module.final_layer_norm = LayerNormQ.from_float(
            module.final_layer_norm, fc1_input_scale
        )
        
        # 转换FFN层
        int8_module.fc1 = W8A8B8O8LinearReLU.from_float(
            module.fc1, fc1_input_scale, fc2_input_scale
        )
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(module.fc2, fc2_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # 自注意力分支（带残差连接）
        residual = hidden_states  # 保存残差连接的输入
        hidden_states = self.self_attn_layer_norm(hidden_states)  # LayerNorm

        # 计算自注意力（INT8量化）
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # 残差连接（注意类型转换）
        residual.add_(hidden_states.to(residual.dtype))

        # 前馈网络分支（带残差连接）
        hidden_states = self.final_layer_norm(residual)  # LayerNorm

        # FFN第一层（INT8，带ReLU）
        hidden_states = self.fc1(hidden_states)

        # FFN第二层（INT8输入，FP32输出）
        hidden_states = self.fc2(hidden_states)

        # 残差连接
        residual.add_(hidden_states.to(residual.dtype))

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Int8OPTDecoder(OPTPreTrainedModel):
    """
    INT8量化的OPT解码器
    
    由多个Int8OPTDecoderLayer组成的Transformer解码器。
    每个层都是INT8量化的，以减少内存占用和加速推理。
    """

    def __init__(self, config):
        """
        初始化INT8解码器
        
        Args:
            config: OPTConfig配置对象
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        # 词嵌入层（保持FP32，因为与lm_head共享权重）
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        
        # 位置嵌入层（保持FP32）
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # 如果词嵌入维度与隐藏层维度不同，需要投影层
        if config.word_embed_proj_dim != config.hidden_size:
            # 输出投影：从隐藏层维度投影到词嵌入维度
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            # 输入投影：从词嵌入维度投影到隐藏层维度
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # 最终LayerNorm（如果需要）
        # 注意：`config._remove_final_layer_norm`仅用于保持向后兼容性
        # 兼容在transformers v4.20.1之前微调的检查点
        # 参见 https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        # 创建多个INT8量化的解码器层
        self.layers = nn.ModuleList(
            [
                Int8OPTDecoderLayer(
                    config.hidden_size, config.num_attention_heads, config.ffn_dim
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 复用OPTDecoder的方法
    get_input_embeddings = OPTDecoder.get_input_embeddings
    set_input_embeddings = OPTDecoder.set_input_embeddings
    # _prepare_decoder_attention_mask = OPTDecoder._prepare_decoder_attention_mask
    old_forward = OPTDecoder.forward  # 保存原始forward方法

    @staticmethod
    def from_float(module, decoder_layer_scales):
        """
        从FP32浮点解码器转换为INT8量化解码器
        
        Args:
            module: FP32的OPTDecoder模块
            decoder_layer_scales: 每层的量化缩放因子字典列表
            
        Returns:
            INT8量化的解码器
        """
        int8_module = Int8OPTDecoder(module.config)
        
        # 保持FP32的嵌入层和投影层（不量化）
        int8_module.embed_tokens = module.embed_tokens
        int8_module.embed_positions = module.embed_positions
        int8_module.project_out = module.project_out
        int8_module.final_layer_norm = module.final_layer_norm
        
        # 逐层转换解码器层为INT8版本
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8OPTDecoderLayer.from_float(
                layer, **decoder_layer_scales[i]
            )
        return int8_module

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        """
        前向传播
        
        为了优化INT8计算性能，输入序列长度会被填充到16的倍数。
        这是为了匹配INT8矩阵乘法的内存对齐要求。
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            head_mask: 注意力头掩码
            past_key_values: 缓存的键值对
            inputs_embeds: 预计算的嵌入（可选）
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式
            
        Returns:
            BaseModelOutputWithPast对象
        """
        # 将输入填充到16的倍数（INT8计算的内存对齐要求）
        input_len = input_ids.shape[1]
        from torch.nn.functional import pad

        if input_len % 16 != 0:
            # OPT的pad token ID是1
            padding_len = 16 - input_len % 16
            input_ids = pad(input_ids, (0, padding_len), value=1)
            if attention_mask is not None:
                attention_mask = pad(attention_mask, (0, padding_len), value=0)
        
        # 调用原始forward方法
        output = self.old_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # 如果进行了填充，将输出截回到原始长度
        if input_len % 16 != 0:
            output.last_hidden_state = output.last_hidden_state[:, :input_len, :]
        return output


class Int8OPTModel(OPTPreTrainedModel):
    """
    INT8量化的OPT模型
    
    封装了INT8量化的解码器，提供完整的模型接口。
    """
    def __init__(self, config: OPTConfig):
        """
        初始化INT8模型
        
        Args:
            config: OPTConfig配置对象
        """
        super().__init__(config)
        self.decoder = Int8OPTDecoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 复用OPTModel的方法
    get_input_embeddings = OPTModel.get_input_embeddings
    set_input_embeddings = OPTModel.set_input_embeddings
    get_decoder = OPTModel.get_decoder
    forward = OPTModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        """
        从FP32浮点模型转换为INT8量化模型
        
        Args:
            module: FP32的OPTModel模块
            decoder_layer_scales: 解码器每层的量化缩放因子字典列表
            
        Returns:
            INT8量化的模型
        """
        int8_module = Int8OPTModel(module.config)
        int8_module.decoder = Int8OPTDecoder.from_float(
            module.decoder, decoder_layer_scales
        )
        return int8_module


class Int8OPTForCausalLM(OPTPreTrainedModel):
    """
    INT8量化的OPT因果语言模型
    
    用于文本生成任务的完整INT8量化模型，包含：
    - INT8量化的模型主体
    - FP32的语言模型头（lm_head），用于输出词汇表概率
    """
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        初始化INT8因果语言模型
        
        Args:
            config: OPTConfig配置对象
        """
        super().__init__(config)
        self.model = Int8OPTModel(config)

        # 语言模型头（保持FP32，因为需要输出精确的概率分布）
        # lm_head的权重会自动与embed_tokens的权重绑定（权重共享）
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        """
        从FP32浮点因果语言模型转换为INT8量化模型
        
        Args:
            module: FP32的OPTForCausalLM模块
            decoder_layer_scales: 解码器每层的量化缩放因子字典列表
            
        Returns:
            INT8量化的因果语言模型
        """
        int8_module = Int8OPTForCausalLM(module.config)
        int8_module.model = Int8OPTModel.from_float(module.model, decoder_layer_scales)
        # 保持FP32的lm_head（不量化）
        int8_module.lm_head = module.lm_head
        return int8_module

    # 复用OPTForCausalLM的所有方法
    get_input_embeddings = OPTForCausalLM.get_input_embeddings
    set_input_embeddings = OPTForCausalLM.set_input_embeddings
    get_output_embeddings = OPTForCausalLM.get_output_embeddings
    set_output_embeddings = OPTForCausalLM.set_output_embeddings
    set_decoder = OPTForCausalLM.set_decoder
    get_decoder = OPTForCausalLM.get_decoder
    forward = OPTForCausalLM.forward
    prepare_inputs_for_generation = OPTForCausalLM.prepare_inputs_for_generation
    _reorder_cache = OPTForCausalLM._reorder_cache
