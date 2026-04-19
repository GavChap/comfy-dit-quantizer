import argparse
import json
import os
import torch
import comfy_kitchen as ck
from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX
from safetensors.torch import safe_open, save_file
import utils
from utils import print_layer_metrics, print_layer_header

QUANTIZABLE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
#TODO: int8s are not clear.
ALLOWED_QTYPES = {"float8_e4m3fn", "float8_e5m2", "nvfp4", "mxfp8", "int8_tensorwise", "int8_rowwise"}
DEFAULT_BLOCK_NAMES = ["block", "transformer", "layer", "model.diffusion_model"]

device = utils.get_device()


def parse_args():
    p = argparse.ArgumentParser(
        prog="quantize.py",
        description="Quantize safetensors weights with rule-based policies.",
    )
    p.add_argument("json", nargs="?", help="Quant config JSON path")
    p.add_argument("src", nargs="*", help="Source safetensors path")
    p.add_argument("dst", nargs="?", help="Target safetensors path")
    p.add_argument("-d", "--downcast-fp32", choices=("fp16", "bf16"), default=None, metavar="{fp16,bf16}",
                   help="Cast fp32 tensors to the selected dtype (default: keep FP32).")
    p.add_argument("-m", "--method", choices=("amax", "mse", "percentile"), default="mse", metavar="{amax, mse, percentile}",
                   help="Set calibration method (default: mse).")
    p.add_argument("-n", "--n-samples", default=None, type=int, help="num of samples for calibration method")
    p.add_argument("-q", "--quiet", action="store_true", help="no verbose.")
    p.add_argument("-v", "--verbose", action="store_true", help="detailed debugging info.")
    p.add_argument("-t", "--test", action="store_true", help="does not save output")
    p.add_argument("-s", "--stochastic", action="store_true", help="use stochastic rounding (INT8 currently supported)")
    p.add_argument("-f", "--dense-search", action="store_true", help="use high-resolution dense scanning for MSE (NVFP4 supported)")
    p.add_argument("-l", "--list", action="store_true", help="List all layers and their sizes, then exit.")
    return p.parse_args()

def list_layers(src_files, block_names=DEFAULT_BLOCK_NAMES, verbose=False):
    total_params = 0
    total_size_mb = 0
    quantizable_layers = []
    other_layers = []

    for f_path in src_files:
        if not os.path.exists(f_path):
            print(f"File not found: {f_path}")
            continue
        
        print(f"Scanning {f_path}...")
        state_dict = load_file(f_path)
        
        for key, tensor in state_dict.items():
            internal_key = key
            if internal_key.startswith("model.diffusion_model."):
                internal_key = internal_key[len("model.diffusion_model."):]
            elif internal_key.startswith("model."):
                internal_key = internal_key[len("model."):]
            
            num_params = tensor.numel()
            size_mb = (num_params * tensor.element_size()) / (1024 * 1024)
            
            total_params += num_params
            total_size_mb += size_mb
            
            is_weight = internal_key.endswith(".weight")
            in_block = any(b in internal_key for b in block_names)
            is_supported_dtype = tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES
            # Check for quantizable linear/conv layers (typically ndim 2 or 4 for conv)
            # The original script uses ndim == 2
            is_quantizable = is_weight and in_block and is_supported_dtype and tensor.ndim == 2
            
            info = {
                "key": key,
                "shape": list(tensor.shape),
                "params": num_params,
                "size_mb": size_mb,
                "dtype": str(tensor.dtype).partition(".")[2]
            }
            
            if is_quantizable:
                quantizable_layers.append(info)
            else:
                other_layers.append(info)

    # Sort quantizable layers by size (params)
    quantizable_layers.sort(key=lambda x: x['params'], reverse=True)
    
    print("\n" + "="*120)
    print(f"{'QUANTIZABLE LAYERS (Sorted by size)':^120}")
    print("="*120)
    print(f"{'Layer Name':<70} {'Shape':<20} {'Params':<15} {'Size (MB)':<10}")
    print("-"*120)
    
    q_params_total = 0
    q_size_total = 0
    for info in quantizable_layers:
        print(f"{info['key']:<70} {str(info['shape']):<20} {info['params']:<15,} {info['size_mb']:<10.2f}")
        q_params_total += info['params']
        q_size_total += info['size_mb']

    print("-"*120)
    print(f"Total Quantizable: {len(quantizable_layers)} layers, {q_params_total:,} params, {q_size_total:.2f} MB")
    
    if verbose:
        print("\n" + "="*120)
        print(f"{'OTHER LAYERS':^120}")
        print("="*120)
        for info in other_layers:
            print(f"{info['key']:<70} {str(info['shape']):<20} {info['params']:<15,} {info['size_mb']:<10.2f}")

    print("\n" + "="*120)
    print(f"{'SUMMARY':^120}")
    print("="*120)
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Total Model Size:       {total_size_mb:.2f} MB")
    if total_params > 0:
        print(f"Quantizable Portion:    {q_params_total/total_params*100:.2f}% of params, {q_size_total/total_size_mb*100:.2f}% of size")
    
    print("\n[RECOMMENDATION]")
    print("Layers best for quantization are those with high parameter counts in the 'QUANTIZABLE LAYERS' list.")
    print("Quantizing the top layers will yield the most significant reduction in model size.")
    if quantizable_layers:
        top_5_percent = sum(l['params'] for l in quantizable_layers[:5])
        if total_params > 0:
            print(f"The top 5 largest layers alone account for {top_5_percent/total_params*100:.2f}% of the total model parameters.")

def quantize_weight(weight, key, quantized_state_dict, quantization_layers, qtype, qformat, method, n_samples, stochastic=False, dense_search=False, verbose=True):
    layer_name = key[:-7]
    
    if qtype == "nvfp4":
        if method == "mse":
            weight_scale_2 = utils.scale_mse_nvfp4(weight, n_samples=n_samples, dense_search=dense_search)
        else:
            weight_scale_2 = utils.scale_amax_nvfp4(weight)
        with ck.use_backend("triton"): # triton supports conversion from fp32
            weight_quantized, weight_scale = ck.quantize_nvfp4(weight, weight_scale_2)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale_2, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.cpu()
    elif qtype == "mxfp8":
        orig_dtype = weight.dtype
        orig_shape = tuple(weight.shape)
        with ck.use_backend("triton"): # triton supports conversion from fp32
            weight_quantized, weight_scale = ck.quantize_mxfp8(weight)
        if verbose: print(layer_name) # TODO: impl. later
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.view(torch.uint8).cpu()
    elif qtype == "int8_tensorwise":
        if method == "mse":
          weight_scale = utils.scale_mse_int8(weight, n_samples=n_samples)
        else:
          weight_scale = utils.scale_amax_int8(weight)
        weight_quantized = utils.quantize_per_tensor_int8(weight, weight_scale, stochastic=stochastic)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
    elif qtype == "int8_rowwise":
        # currently, it doesn't support mse
        if method == "percentile":
            weight_scales = utils.scale_rowwise_percentile_int8(weight)
        else:
            weight_scales = utils.scale_rowwise_amax_int8(weight)
        weight_quantized = utils.quantize_rowwise_int8(weight, weight_scales, stochastic=stochastic)
        if verbose: print(layer_name) # TODO: impl. later
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scales.unsqueeze(-1).cpu()
    elif qtype == "float8_e5m2":
        pass
        # todo
    else: # fp8 e4m3
        if method == "mse":
            weight_scale = utils.scale_mse_fp8(weight, n_samples=n_samples)
        else:
            weight_scale = utils.scale_amax_fp8(weight)
        weight_quantized = ck.quantize_per_tensor_fp8(weight, weight_scale)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    ## The format is not clear at the moment.
    # "format": qtype                                          - It is definitely necessary.
    # "full_precision_matrix_mult": False                      - ComfyUI use this
    # "block_size": 32                                         - Kijai's mxfp8 quant model has this.
    # "orig_dtype": "torch.bfloat16", "orig_shape": orig_shape - QuantOps uses these.

    if qtype == "mxfp8": # At the moment, I'll follow Kijai's.
        quant_info = { "format": qtype, "block_size": 32, "full_precision_matrix_mult": False }
    else:
        quant_info = { "format": qtype, "full_precision_matrix_mult": False }

    if qformat == "comfy_quant":
        quantized_state_dict[f"{layer_name}.comfy_quant"] = torch.tensor(
                list(json.dumps(quant_info).encode("utf-8")), dtype=torch.uint8)
    else: # 1.0
        quantization_layers[layer_name] = quant_info

def store_with_optional_downcast(tensor, key, quantized_state_dict, cast_to, verbose=True):
    if tensor.dtype == torch.float32 and cast_to != None:
        casted_weight = tensor.to(dtype=cast_to)
        quantized_state_dict[key] = casted_weight.cpu()

        if verbose and key.endswith(".weight"):
            layer_name = key[:-7]
            print_layer_metrics(layer_name, tensor, casted_weight)
    else:
        quantized_state_dict[key] = tensor.cpu()

def first_matching_qtype_for_key(key, rules, verbose=False):
    for i, r in enumerate(rules):
        matches = r.get("match", [])
        for p in matches:
            if p in key:
                qtype = r.get("policy")
                if verbose:
                    print(f"  [DEBUG] Match found: rule #{i} (pattern '{p}') -> {qtype}")
                return qtype if qtype in ALLOWED_QTYPES else None
    if verbose:
        print(f"  [DEBUG] No match found for '{key}' in {len(rules)} rules.")
    return None

def main():
    args = parse_args()
    
    if args.list:
        src_files = [args.json] if args.json else []
        if args.src: src_files.extend(args.src)
        if args.dst: src_files.append(args.dst)
        
        if not src_files:
            print("Error: Provide at least one safetensors file to list.")
            return
            
        block_names = DEFAULT_BLOCK_NAMES
        if args.json and args.json.endswith(".json"):
            with open(args.json, "r", encoding="utf-8") as f:
                config = json.load(f)
                block_names = config.get("block_names", DEFAULT_BLOCK_NAMES)
                # If json was indeed a config, don't treat it as a src file
                src_files = src_files[1:]

        list_layers(src_files, block_names, verbose=args.verbose)
        return

    # Redistribute arguments for quantization mode since they are now optional in argparse
    if args.dst is None and len(args.src) > 0:
        args.dst = args.src.pop()
    
    if not args.json or not args.src or not args.dst:
        print("Usage error: [config.json] [source.safetensors ...] [destination.safetensors] are required for quantization.")
        print("Use --list if you only want to inspect a model.")
        return

    cast_to = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.downcast_fp32, None)
    assert args.json and ".json" in args.json, f"{args.json} is not .json file."
    with open(args.json, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Default to comfy_quant if not specified, as ops.py expects it
    qformat = config.get("format", "comfy_quant")
    block_names = config.get("block_names", DEFAULT_BLOCK_NAMES)
    rules = config.get("rules", [])
    
    # Optional prefix stripping for merged models
    strip_prefix = config.get("strip_prefix", "")
    rules = config.get("rules", [])
    if rules and rules[0].get("match") and len(rules[0]["match"]) > 0:
        if "model." in rules[0]["match"][0]: # Heuristic
             if args.verbose: print(f"[DEBUG] Heuristic triggered: resetting strip_prefix")
             strip_prefix = "" 

    if args.verbose:
        print(f"[DEBUG] Device: {device}")
        print(f"[DEBUG] Config rules: {len(rules)}")
        print(f"[DEBUG] Block names: {block_names}")

    quantized_state_dict, quantization_layers, merged_metadata = {}, {}, {}
    from safetensors import safe_open
    
    for f in args.src:
        print(f"Opening {f}...")
        if not args.quiet: print_layer_header()

        #state_dict = load_file(f)
        with safe_open(f, framework="pt", device="cpu") as f:
            m = f.metadata()
            if m:
                merged_metadata.update(m)
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        for key, tensor in state_dict.items():
            original_key = key
            # Handle common prefixes in merged models
            if key.startswith("model.diffusion_model."):
                key = key[len("model.diffusion_model."):]
            elif key.startswith("model."):
                key = key[len("model."):]

            if not (any(b in key for b in block_names) and key.endswith(".weight")
                    and tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES and tensor.ndim == 2):
                if args.verbose:
                    reason = []
                    if not any(b in key for b in block_names): reason.append("not in block_names")
                    if not key.endswith(".weight"): reason.append("not a weight")
                    if tensor.dtype not in QUANTIZABLE_WEIGHT_DTYPES: reason.append(f"unsupported dtype {tensor.dtype}")
                    if tensor.ndim != 2: reason.append(f"ndim={tensor.ndim} (expected 2)")
                    print(f"  [DEBUG] Skipping {original_key}: {', '.join(reason)}")
                store_with_optional_downcast(tensor, original_key, quantized_state_dict, cast_to, verbose=not args.quiet)
                continue

            if args.verbose: print(f"[DEBUG] Processing {original_key} (internal: {key}) shape: {list(tensor.shape)}")
            qtype = first_matching_qtype_for_key(key, rules, verbose=args.verbose)
            if qtype is None:
                store_with_optional_downcast(tensor, original_key, quantized_state_dict, cast_to, verbose=not args.quiet)
            else:
                if not args.quiet:
                    print(f"Quantizing {original_key} as {qtype} (Internal name: {key})")
                quantize_weight(tensor.to(device), original_key, quantized_state_dict, quantization_layers, qtype, qformat, args.method, args.n_samples, stochastic=args.stochastic, dense_search=args.dense_search, verbose=not args.quiet)

    if not args.test:
        if qformat != "comfy_quant":
            merged_metadata["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": quantization_layers})
        save_file(quantized_state_dict, args.dst, metadata=merged_metadata)
        total_bytes = os.path.getsize(args.dst)
        print(f"Output: {args.dst} ({round(total_bytes / (1024**3), 2)}GB)")

if __name__ == "__main__":
    main()
