"""Test TorchLens support for ResNet and ViT model families (using replay method)"""
import time
import torch
import torchvision
import torchlens as tl


def prepare_replay_graph(net_list):
    """Prepare all layers for fast replay and build label2idx mapping.
    
    Args:
        net_list: List of TensorLogEntry from model_history.layer_list
        
    Returns:
        label2idx: Dict mapping layer_label to index
    """
    label2idx = {}
    for i, layer in enumerate(net_list):
        label2idx[layer.layer_label] = i
        if layer.func_applied_name != 'none':
            layer.prepare_replay()
    return label2idx


def model_log_forward_fast(net_list, x, label2idx=None):
    """Optimized layer-by-layer forward using replay_fast.
    
    Args:
        net_list: List of TensorLogEntry from model_history.layer_list
        x: Input tensor
        label2idx: Optional precomputed label to index mapping (from prepare_replay_graph).
                   If None, will be computed internally.
        
    Returns:
        Output tensor from the forward pass.
    """
    if label2idx is None:
        label2idx = {layer.layer_label: i for i, layer in enumerate(net_list)}
    
    for layer in net_list:
        func_name = layer.func_applied_name
        layer_type = layer.layer_type
        
        if func_name == 'none':
            layer.tensor_contents = x if layer_type == 'input' else (layer.tensor_contents if layer_type == 'buffer' else None)
            continue

        x_in, buffer_in = [], []
        op_num = layer.operation_num
        for plabel in layer.parent_layers:
            p = net_list[label2idx[plabel]]
            if p.layer_type == 'buffer':
                buffer_in.append(p.tensor_contents)
            else:
                tc = p.tensor_contents
                x_in.append(x if tc is None and p.operation_num == op_num - 1 else tc)
                if p.layer_type not in ('input', 'buffer') and label2idx[p.child_layers[-1]] <= label2idx[layer.layer_label]:
                    p.tensor_contents = None

        x = layer.replay_fast(x_in, buffer_in)
        layer.tensor_contents = None if layer.child_layers and net_list[label2idx[layer.child_layers[-1]]].operation_num <= op_num + 1 else x
    return x


def test_model(model_name, model, input_tensor, num_runs=5):
    """Test a single model - validate using replay method and measure timing"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    try:
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Measure original model timing
        t0 = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                res_original = model(input_tensor)
        time_original = (time.perf_counter() - t0) / num_runs * 1000  # ms
        
        # Log forward pass with torchlens (only once)
        model_history = tl.log_forward_pass(model, input_tensor, vis_opt='none', save_function_args=True)
        layer_list = model_history.layer_list
        
        # Prepare fast replay (also returns label2idx)
        label2idx = prepare_replay_graph(layer_list)
        
        # Warmup replay
        _ = model_log_forward_fast(layer_list, input_tensor, label2idx)
        
        # Measure replay_fast timing
        t0 = time.perf_counter()
        for _ in range(num_runs):
            res_replay = model_log_forward_fast(layer_list, input_tensor, label2idx)
        time_replay = (time.perf_counter() - t0) / num_runs * 1000  # ms
        
        match = torch.allclose(res_replay, res_original, atol=1e-5)
        ratio = time_replay / time_original
        
        print(f"Layers: {len(layer_list)}")
        print(f"Output shape: {res_replay.shape}")
        print(f"Original model time: {time_original:.2f} ms")
        print(f"Replay time: {time_replay:.2f} ms")
        print(f"Replay/Original: {ratio:.2f}x")
        print(f"Replay output match: {'✓ PASS' if match else '✗ FAIL'}")
        
        if not match:
            diff = (res_replay - res_original).abs().max().item()
            print(f"Max difference: {diff}")
        
        return match, time_original, time_replay
    except Exception as e:
        import traceback
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False, 0, 0


def main():
    # Standard input
    x_224 = torch.rand(1, 3, 224, 224)
    
    results = {}
    times = {}
    
    # ResNet family
    print("\n" + "="*60)
    print("ResNet Model Family Test")
    print("="*60)
    
    resnet_models = [
        ("ResNet-18", torchvision.models.resnet18(weights=None)),
        ("ResNet-50", torchvision.models.resnet50(weights=None)),
        ("ResNet-101", torchvision.models.resnet101(weights=None)),
    ]
    
    for name, model in resnet_models:
        match, t_orig, t_replay = test_model(name, model, x_224)
        results[name] = match
        times[name] = (t_orig, t_replay)
    
    # ViT family
    print("\n" + "="*60)
    print("ViT Model Family Test")
    print("="*60)
    
    vit_models = [
        ("ViT-B/16", torchvision.models.vit_b_16(weights=None)),
        ("ViT-B/32", torchvision.models.vit_b_32(weights=None)),
    ]
    
    for name, model in vit_models:
        match, t_orig, t_replay = test_model(name, model, x_224)
        results[name] = match
        times[name] = (t_orig, t_replay)
    
    # Other models
    print("\n" + "="*60)
    print("Other Models Test")
    print("="*60)
    
    other_models = [
        ("DenseNet-121", torchvision.models.densenet121(weights=None)),
        ("MobileNetV3", torchvision.models.mobilenet_v3_small(weights=None)),
        ("Swin-T", torchvision.models.swin_t(weights=None)),
    ]
    
    for name, model in other_models:
        match, t_orig, t_replay = test_model(name, model, x_224)
        results[name] = match
        times[name] = (t_orig, t_replay)
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"\n{'Model':<20} {'Original(ms)':<12} {'Replay(ms)':<12} {'Ratio':<8}")
    print("-" * 52)
    for name in results:
        t_orig, t_replay = times[name]
        ratio = t_replay / t_orig if t_orig > 0 else 0
        status = "✓" if results[name] else "✗"
        print(f"{status} {name:<18} {t_orig:<12.2f} {t_replay:<12.2f} {ratio:.2f}x")


if __name__ == "__main__":
    main()
