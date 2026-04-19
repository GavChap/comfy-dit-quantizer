import torch

def quantize_per_tensor_int8(x, scale, stochastic=False):
    x_scaled = x.float().mul(1.0 / scale)
    if stochastic:
        x_scaled.add_(torch.rand_like(x_scaled)).floor_()
    else:
        x_scaled.round_()
    return x_scaled.clamp_(-128.0, 127.0).to(torch.int8)

def dequantize_per_tensor_int8(x, scale):
    return x.float() * scale

def quantize_rowwise_int8(x, scales, stochastic=False):
    x_float = x.float()
    quantized = torch.zeros_like(x, dtype=torch.int8)

    for i in range(scales.numel()):
        row = x_float[i] * (1.0 / scales[i])
        if stochastic:
            row.add_(torch.rand_like(row)).floor_()
        else:
            row.round_()
        quantized[i] = torch.clamp(row, -128, 127).to(torch.int8)

    return quantized

def dequantize_rowwise_int8(x, scales):
    x_float = x.float()
    dequantized = torch.zeros_like(x_float)

    for i in range(scales.numel()):
        dequantized[i] = x_float[i] * scales[i]

    return dequantized
