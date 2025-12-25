"""测试 TorchLens 对 ResNet 和 ViT 系列模型的支持（使用 replay 方式）"""
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
    """测试单个模型 - 使用 replay 方式验证，并统计用时"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    try:
        model.eval()
        
        # 预热
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 测量原始模型用时
        t0 = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                res_original = model(input_tensor)
        time_original = (time.perf_counter() - t0) / num_runs * 1000  # ms
        
        # 使用 torchlens 记录前向传播（只记录一次）
        model_history = tl.log_forward_pass(model, input_tensor, vis_opt='none', save_function_args=True)
        layer_list = model_history.layer_list
        
        # 准备 fast replay（同时返回 label2idx）
        label2idx = prepare_replay_graph(layer_list)
        
        # 预热 replay
        _ = model_log_forward_fast(layer_list, input_tensor, label2idx)
        
        # 测量 replay_fast 用时
        t0 = time.perf_counter()
        for _ in range(num_runs):
            res_replay = model_log_forward_fast(layer_list, input_tensor, label2idx)
        time_replay = (time.perf_counter() - t0) / num_runs * 1000  # ms
        
        match = torch.allclose(res_replay, res_original, atol=1e-5)
        ratio = time_replay / time_original
        
        print(f"层数: {len(layer_list)}")
        print(f"输出形状: {res_replay.shape}")
        print(f"原始模型用时: {time_original:.2f} ms")
        print(f"Replay 用时: {time_replay:.2f} ms")
        print(f"Replay/Original: {ratio:.2f}x")
        print(f"Replay 输出一致性: {'✓ 通过' if match else '✗ 失败'}")
        
        if not match:
            diff = (res_replay - res_original).abs().max().item()
            print(f"最大差异: {diff}")
        
        return match, time_original, time_replay
    except Exception as e:
        import traceback
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False, 0, 0


def main():
    # 标准输入
    x_224 = torch.rand(1, 3, 224, 224)
    
    results = {}
    times = {}
    
    # ResNet 系列
    print("\n" + "="*60)
    print("ResNet 系列模型测试")
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
    
    # ViT 系列
    print("\n" + "="*60)
    print("ViT 系列模型测试")
    print("="*60)
    
    vit_models = [
        ("ViT-B/16", torchvision.models.vit_b_16(weights=None)),
        ("ViT-B/32", torchvision.models.vit_b_32(weights=None)),
    ]
    
    for name, model in vit_models:
        match, t_orig, t_replay = test_model(name, model, x_224)
        results[name] = match
        times[name] = (t_orig, t_replay)
    
    # 其他模型
    print("\n" + "="*60)
    print("其他模型测试")
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
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    print(f"\n{'模型':<20} {'原始(ms)':<12} {'Replay(ms)':<12} {'比率':<8}")
    print("-" * 52)
    for name in results:
        t_orig, t_replay = times[name]
        ratio = t_replay / t_orig if t_orig > 0 else 0
        status = "✓" if results[name] else "✗"
        print(f"{status} {name:<18} {t_orig:<12.2f} {t_replay:<12.2f} {ratio:.2f}x")


if __name__ == "__main__":
    main()
