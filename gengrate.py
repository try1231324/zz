import torch
import os
import numpy as np
import math

# 读取 .pth 文件路径
pth_file = '/cwj/training-mixed-precision-quantized-networks/results/Imagenet/mobilenet_cifar100_4bit_icn160PLPACT2/full_int_model_with_quant_params.pth'
save_dir = 'layer_h_files3'

os.makedirs(save_dir, exist_ok=True)

# 设置 Z_w 偏移为 8（权重导出脚本里用到，这里只是生成调用 C，不再使用）
Z_w = 8

# 第一层输入尺寸（正方形）
FIRST_DIM = 160

# 加载 pth
data = torch.load(pth_file, map_location='cpu')
all_layers_params = data['quant_params']

def as_pair(x, default_when_none=None):
    """把 x 规范成 (h, w)。x 可以是 int / tuple / list / None。"""
    if x is None:
        if default_when_none is None:
            return (None, None)
        return (default_when_none, default_when_none)
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return (int(x[0]), int(x[0]))
        elif len(x) >= 2:
            return (int(x[0]), int(x[1]))
        else:
            return (None, None)
    # int
    return (int(x), int(x))

def parse_padding(padding):
    """解析 padding 为 (left, right, top, bottom)。支持 None / int / (h,w) / (l,r,t,b)"""
    if padding is None:
        return (0, 0, 0, 0)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h, pad_w = int(padding[0]), int(padding[1])
            return (pad_w, pad_w, pad_h, pad_h)
        elif len(padding) == 4:
            # 可能是 (left, right, top, bottom) 或 (top, left, bottom, right) 的其他框架风格
            # 这里假设你导出的就是 (pad_w, pad_w, pad_h, pad_h) 或 torch 的 (pad_h, pad_w)
            l, r, t, b = padding
            return (int(l), int(r), int(t), int(b))
        else:
            # 不支持的就当 0
            return (0, 0, 0, 0)
    # 单值
    v = int(padding)
    return (v, v, v, v)

def generate_layer_files():
    # ========= 预扫描，计算缓冲区最大需求 =========
    prev_dim = FIRST_DIM
    max_feature_size = 0
    max_im2col_size = 0
    max_bufferb_size = 0

    first_conv_in_ch = None

    for i, layer in enumerate(all_layers_params):
        conv = layer.get('quant_conv')
        if not conv:
            continue

        in_channels = conv.get('in_channels')
        out_channels = conv.get('out_channels')
        kernel_size = conv.get('kernel_size')
        stride = conv.get('stride')
        padding = conv.get('padding')
        groups = conv.get('groups', 1) or 1

        # 记录第一层的 in_channels 用于首帧输入 buffer 需求
        if first_conv_in_ch is None and in_channels is not None:
            first_conv_in_ch = int(in_channels)

        # 参数健壮化
        kh, kw = as_pair(kernel_size)
        if kh is None:
            # 缺关键参数，跳过
            continue
        sh, sw = as_pair(stride, default_when_none=1)
        left_pad, right_pad, top_pad, bottom_pad = parse_padding(padding)

        # 输出特征尺寸（正方图）
        dim_out = ((prev_dim + top_pad + bottom_pad - kh) // sh) + 1

        # 输出特征字节数（2 个 4bit 合 1 字节）
        feature_size = (dim_out * dim_out * int(out_channels) + 1) // 2
        if feature_size > max_feature_size:
            max_feature_size = feature_size

        # bufferA: 2 * ch_im_in * k * k   （int16_t）
        im2col_size = 2 * int(in_channels) * kh * kh
        if im2col_size > max_im2col_size:
            max_im2col_size = im2col_size

        # bufferB：先给一个上界（没用也不怕）
        bufferb_size = 2 * int(out_channels) * kh * kh
        if bufferb_size > max_bufferb_size:
            max_bufferb_size = bufferb_size

        prev_dim = dim_out

    # 把首帧输入也纳入 feature 的上限（防止第一层输入比后面任一层输出还大）
    if first_conv_in_ch is not None:
        first_in_size = (FIRST_DIM * FIRST_DIM * first_conv_in_ch + 1) // 2
        if first_in_size > max_feature_size:
            max_feature_size = first_in_size

    # ========= 生成 C 代码 =========
    c = []
    c.append('#include <stddef.h>\n#include <stdint.h>\n')
    c.append('#include "merged_layers.h"\n\n')

    c.append(f"#define MAX_FEATURE_SIZE {max_feature_size}\n")
    c.append(f"#define MAX_IM2COL_SIZE {max_im2col_size}\n")
    c.append(f"#define MAX_BUFFERB_SIZE {max_bufferb_size}\n\n")

    c.append("static uint8_t  buffer0[MAX_FEATURE_SIZE];\n")
    c.append("static uint8_t  buffer1[MAX_FEATURE_SIZE];\n")
    c.append("static int16_t  bufferA[MAX_IM2COL_SIZE];\n")
    c.append("static uint8_t  bufferB[MAX_BUFFERB_SIZE];\n\n")

    c.append("void invoke_layers(void) {\n")

    # 双缓冲
    prev_dim = FIRST_DIM
    input_buf = "buffer0"
    output_buf = "buffer1"

    for i, layer in enumerate(all_layers_params):
        conv = layer.get('quant_conv')
        if not conv:
            continue

        # 量化偏移，没给就 0
        Z_in_val  = int(conv.get('Z_IN', 0))
        Z_out_val = int(conv.get('Z_OUT', 0))

        in_channels  = conv.get('in_channels')
        out_channels = conv.get('out_channels')
        kernel_size  = conv.get('kernel_size')
        stride       = conv.get('stride')
        padding      = conv.get('padding')
        groups       = conv.get('groups', 1) or 1

        if in_channels is None or out_channels is None or kernel_size is None:
            # 缺关键参数，这层跳过
            continue

        in_channels  = int(in_channels)
        out_channels = int(out_channels)

        kh, kw = as_pair(kernel_size)
        sh, sw = as_pair(stride, default_when_none=1)
        left_pad, right_pad, top_pad, bottom_pad = parse_padding(padding)

        # 输出尺寸（正方）
        dim_out = ((prev_dim + top_pad + bottom_pad - kh) // sh) + 1

        # 哪个函数
        if int(groups) == in_channels:
            c.append(f"  // Layer {i} depthwise convolution\n")
            c.append("  arm_depthwise_separable_conv_HWC_u4_u4_u4(\n")
        else:
            c.append(f"  // Layer {i} convolution\n")
            c.append("  arm_convolve_HWC_int4_u4_int4(\n")

        # bias 符号，不存在则传 NULL
        bias_exists = conv.get('bias') is not None
        bias_sym = f"BIAS_{i}" if bias_exists else "NULL"

        # m_zero / n_zero：当前用 0/0（你的头文件里是数组，函数签名要标量；等你确定聚合策略再替换）
        m_zero_sym = f"M_ZERO_{i}"
        n_zero_sym = f"N_ZERO_{i}"


        # 生成参数行
        c.append(
            f"    {input_buf}, {prev_dim}, {in_channels}, WEIGHT_{i}, {out_channels}, {kh}, "
            f"{left_pad}, {right_pad}, {top_pad}, {bottom_pad}, {sh}, "
            f"{bias_sym}, {output_buf}, {dim_out}, "
            f"{Z_in_val}, Z_W_{i}, {Z_out_val}, {m_zero_sym}, {n_zero_sym}, bufferA, bufferB);\n\n"
        )

        # 交替缓冲区 + 更新尺寸
        input_buf, output_buf = output_buf, input_buf
        prev_dim = dim_out
    c.append("}\n")
    

    with open('generated_layers.c', 'w') as f:
        f.write("".join(c))

    print("✅ C 侧包装已生成：generated_layers.c")
    print(f"   MAX_FEATURE_SIZE = {max_feature_size} bytes")
    print(f"   MAX_IM2COL_SIZE  = {max_im2col_size} int16 elements")
    print(f"   MAX_BUFFERB_SIZE = {max_bufferb_size} bytes")

generate_layer_files()

# import torch
# import os
# import numpy as np

# # 读取 .pth 文件路径
# pth_file = '/cwj/training-mixed-precision-quantized-networks/results/Imagenet/mobilenet_cifar100_4bit_icn160PLPACT2/full_int_model_with_quant_params.pth'
# save_dir = 'layer_h_files3'

# os.makedirs(save_dir, exist_ok=True)

# Z_w = 8
# FIRST_DIM = 160
# data = torch.load(pth_file, map_location='cpu')
# all_layers_params = data['quant_params']

# def combine_4bit_to_8bit(weight_4bit):
#     weight_4bit = weight_4bit.flatten()
#     combined_weight = []
#     for i in range(0, len(weight_4bit), 2):
#         low_4bit = weight_4bit[i] & 0x0F
#         high_4bit = (weight_4bit[i + 1] & 0x0F) << 4
#         combined_weight.append(low_4bit | high_4bit)
#     return np.array(combined_weight, dtype=np.uint8)

# def parse_padding(padding):
#     if isinstance(padding, (tuple, list)):
#         if len(padding) == 2:
#             pad_h, pad_w = padding
#             return pad_w or 0, pad_w or 0, pad_h or 0, pad_h or 0
#         elif len(padding) == 4:
#             return tuple(p or 0 for p in padding)
#         else:
#             raise ValueError(f"Unsupported padding format: {padding}")
#     else:
#         return padding or 0, padding or 0, padding or 0, padding or 0

# def as_pair(value, default=1):
#     if isinstance(value, (tuple, list)):
#         return (value[0] or default, value[1] or default)
#     elif value is None:
#         return default, default
#     else:
#         return value, value

# def generate_layer_files():
#     prev_dim = FIRST_DIM
#     max_feature_size = 0
#     max_im2col_size = 0
#     max_bufferb_size = 0
#     first_conv_in_ch = None
#     max_feature_layer_idx = -1

#     # ========= 预扫描计算最大缓冲区 =========
#     for i, layer in enumerate(all_layers_params):
#         conv = layer.get('quant_conv')
#         if not conv:
#             continue
#         in_channels = conv.get('in_channels')
#         out_channels = conv.get('out_channels')
#         kernel_size = conv.get('kernel_size')
#         stride = conv.get('stride')
#         padding = conv.get('padding')
#         groups = conv.get('groups', 1) or 1

#         if in_channels is None or out_channels is None or kernel_size is None:
#             continue

#         kh, kw = as_pair(kernel_size)
#         sh, sw = as_pair(stride, default=1)
#         left_pad, right_pad, top_pad, bottom_pad = parse_padding(padding)

#         # 这里确保全部都是整数
#         prev_dim_int = prev_dim or 1
#         kh = kh or 1
#         sh = sh or 1
#         dim_out = ((prev_dim_int + top_pad + bottom_pad - kh) // sh) + 1
#         feature_size = (dim_out * dim_out * int(out_channels) + 1) // 2

#         if feature_size > max_feature_size:
#             max_feature_size = feature_size
#             max_feature_layer_idx = i

#         im2col_size = 2 * int(in_channels) * kh * kh
#         if im2col_size > max_im2col_size:
#             max_im2col_size = im2col_size

#         bufferb_size = 2 * int(out_channels) * kh * kh
#         if bufferb_size > max_bufferb_size:
#             max_bufferb_size = bufferb_size

#         prev_dim = dim_out
#         if first_conv_in_ch is None:
#             first_conv_in_ch = int(in_channels)

#     if first_conv_in_ch is not None:
#         first_in_size = (FIRST_DIM * FIRST_DIM * first_conv_in_ch + 1) // 2
#         if first_in_size > max_feature_size:
#             max_feature_size = first_in_size
#             max_feature_layer_idx = -1

#     # ========= 生成 C 文件 =========
#     c_code = []
#     c_code.append('#include <stddef.h>\n#include <stdint.h>\n#include "merged_layers.h"\n\n')
        
#     if max_feature_layer_idx >= 0:

#         c_code.append(f"/* MAX_FEATURE_SIZE = {max_feature_size} bytes, 由第 {max_feature_layer_idx} 层输出计算得出 */\n")
#     else:
#         c_code.append(f"/* MAX_FEATURE_SIZE = {max_feature_size} bytes, 由第一层输入计算得出 */\n")
#     c_code.append(f"#define MAX_FEATURE_SIZE {max_feature_size}\n")
#     c_code.append(f"#define MAX_IM2COL_SIZE {max_im2col_size}\n")
#     c_code.append(f"#define MAX_BUFFERB_SIZE {max_bufferb_size}\n\n")
#     c_code.append(f"static uint8_t  buffer0[MAX_FEATURE_SIZE];\n")
#     c_code.append(f"static uint8_t  buffer1[MAX_FEATURE_SIZE];\n")
#     c_code.append(f"static int16_t  bufferA[MAX_IM2COL_SIZE];\n")
#     c_code.append(f"static uint8_t  bufferB[MAX_BUFFERB_SIZE];\n\n")

#     c_code.append('void invoke_layers(uint8_t* buffer0, uint8_t* buffer1, int16_t* bufferA, uint8_t* bufferB) {\n')

#     prev_dim = FIRST_DIM
#     input_buf = "buffer0"
#     output_buf = "buffer1"

#     for i, layer in enumerate(all_layers_params):
#         conv = layer.get('quant_conv')
#         if not conv:
#             continue

#         Z_in_val = int(conv.get('Z_IN', 0) or 0)
#         Z_out_val = int(conv.get('Z_OUT', 0) or 0)

#         in_channels = conv.get('in_channels')
#         out_channels = conv.get('out_channels')
#         kernel_size = conv.get('kernel_size')
#         stride = conv.get('stride')
#         padding = conv.get('padding')
#         groups = conv.get('groups', 1)

#         if in_channels is None or out_channels is None or kernel_size is None:
#             continue

#         kh, kw = as_pair(kernel_size)
#         sh, sw = as_pair(stride, default=1)
#         left_pad, right_pad, top_pad, bottom_pad = parse_padding(padding)

#         prev_dim_int = prev_dim or 1
#         kh = kh or 1
#         sh = sh or 1
#         dim_out = ((prev_dim_int + top_pad + bottom_pad - kh) // sh) + 1

#         if groups == in_channels:
#             c_code.append(f"  // Layer {i} depthwise convolution\n")
#             c_code.append(f"  arm_depthwise_separable_conv_HWC_u4_u4_u4(\n")
#         else:
#             c_code.append(f"  // Layer {i} convolution\n")
#             c_code.append(f"  arm_convolve_HWC_int4_u4_int4(\n")

#         c_code.append(
#             f"    {input_buf}, {prev_dim_int}, {in_channels}, WEIGHT_{i}, {out_channels}, {kh}, "
#             f"{left_pad}, {right_pad}, {top_pad}, {bottom_pad}, {sh}, "
#             f"BIAS_{i}, {output_buf}, {dim_out}, "
#             f"{Z_in_val}, Z_W_{i}, {Z_out_val}, M_ZERO_{i}, N_ZERO_{i}, bufferA, bufferB);\n\n"
#         )

#         # 双缓冲切换
#         input_buf, output_buf = output_buf, input_buf
#         prev_dim = dim_out

#     c_code.append("}\n")

#     with open('generated_layers.c', 'w') as f:
#         f.write("\n".join(c_code))

#     print("✅ C code generated and saved to 'generated_layers.c'")

# generate_layer_files()

