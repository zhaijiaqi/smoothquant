"""
校准模块 (Calibration Module)

该模块用于在模型量化前收集激活值的统计信息，特别是激活值的尺度(scale)信息。
主要用于SmoothQuant量化方法的前期准备工作，通过统计实际数据上的激活值分布
来确定合适的量化参数。

主要功能：
1. get_act_scales: 收集每个Linear层输入的激活值尺度（每个特征维度的最大值）
2. get_static_decoder_layer_scales: 收集decoder层特定模块的输入输出激活值尺度
"""

import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    收集模型中所有Linear层输入的激活值尺度（activation scales）
    
    该函数通过前向传播钩子(hook)捕获每个Linear层的输入激活值，统计每个特征维度
    的绝对最大值。这些尺度信息用于后续的量化参数计算。
    
    Args:
        model: 待校准的模型
        tokenizer: 用于文本编码的tokenizer
        dataset_path: 校准数据集路径（JSON格式）
        num_samples: 用于校准的样本数量，默认512
        seq_len: 序列最大长度，默认512
    
    Returns:
        act_scales: 字典，键为模块名称，值为该模块输入每个特征维度的最大激活值
                   形状为 [hidden_dim] 的tensor
    """
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备
    act_scales = {}  # 存储每个模块的激活值尺度

    def stat_tensor(name, tensor):
        """
        统计tensor每个特征维度的最大绝对值
        
        将tensor重塑为 [batch*seq_len, hidden_dim] 的形状，然后统计每个特征维度
        在所有样本和位置上的最大绝对值。
        """
        hidden_dim = tensor.shape[-1]  # 获取隐藏层维度
        # 将tensor重塑为 [batch*seq_len, hidden_dim] 并取绝对值
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        # 计算每个特征维度的最大值，并移到CPU
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        # 与之前统计的最大值取较大者（跨样本累积最大值）
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        """
        前向传播钩子函数，在每次前向传播时调用
        
        捕获Linear层的输入激活值并进行统计。
        """
        if isinstance(x, tuple):
            x = x[0]  # 如果输入是元组，取第一个元素
        stat_tensor(name, x)

    # 为所有Linear层注册前向传播钩子
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # 加载并打乱校准数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    # 遍历样本进行前向传播，收集激活值统计信息
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)  # 前向传播，触发钩子函数

    # 移除所有钩子
    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    """
    获取decoder层的静态激活值尺度，用于静态量化
    
    该函数专门针对decoder架构的模型（如GPT、LLaMA等），收集每个decoder层中
    关键模块（注意力层、前馈网络）的输入和输出激活值的最大绝对值。这些信息
    用于计算静态量化时的scale参数。
    
    与get_act_scales的区别：
    - 本函数统计的是整个tensor的最大值（标量），而不是每个特征维度的最大值
    - 同时统计输入和输出的激活值尺度
    - 专门提取decoder层特定模块的尺度信息
    
    Args:
        model: 待校准的decoder模型
        tokenizer: 用于文本编码的tokenizer
        dataset_path: 校准数据集路径（JSON格式）
        num_samples: 用于校准的样本数量，默认512
        seq_len: 序列最大长度，默认512
    
    Returns:
        decoder_layer_scales: 列表，每个元素是一个字典，包含该decoder层的各种scale
                             字典键包括：
                             - attn_input_scale: 注意力层输入scale
                             - q_output_scale: Q投影输出scale
                             - k_output_scale: K投影输出scale
                             - v_output_scale: V投影输出scale
                             - out_input_scale: 输出投影输入scale
                             - fc1_input_scale: 第一个全连接层输入scale
                             - fc2_input_scale: 第二个全连接层输入scale
                             所有scale都已除以127（INT8量化的最大值）
        act_dict: 完整的激活值字典，包含所有Linear层的输入输出最大值
    """
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 存储每个模块的输入和输出激活值最大值
    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        """
        前向传播钩子函数，统计模块的输入和输出激活值最大值
        
        同时记录输入和输出的绝对最大值，用于后续计算量化scale。
        """
        if isinstance(x, tuple):
            x = x[0]  # 如果输入是元组，取第一个元素
        # 统计输入激活值的最大绝对值（整个tensor的最大值）
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]  # 如果输出是元组，取第一个元素
        # 统计输出激活值的最大绝对值（整个tensor的最大值）
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    # 为所有Linear层注册前向传播钩子
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    # 加载并打乱校准数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    
    # 遍历样本进行前向传播，收集激活值统计信息
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)  # 前向传播，触发钩子函数
        # 显示平均输入scale作为进度提示
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    
    # 移除所有钩子
    for hook in hooks:
        hook.remove()

    # 提取每个decoder层的特定模块scale信息
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        # 计算各个模块的量化scale（除以127，因为INT8的范围是[-128, 127]）
        # 注意力层输入scale（使用q_proj的输入作为代表）
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
        )
        # Q投影输出scale
        scale_dict["q_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
        )
        # K投影输出scale
        scale_dict["k_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
        )
        # V投影输出scale
        scale_dict["v_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
        )
        # 输出投影输入scale
        scale_dict["out_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
        )
        # 第一个全连接层输入scale
        scale_dict["fc1_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
        )
        # 第二个全连接层输入scale
        scale_dict["fc2_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
