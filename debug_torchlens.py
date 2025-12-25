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


def format_flops(flops):
    """Format FLOPs number to human readable string."""
    if flops is None:
        return "N/A"
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    else:
        return str(int(flops))


def print_flops_summary(layer_list):
    """Print FLOPs summary for all layers."""
    total_forward = 0
    total_backward = 0
    
    print("\n" + "="*80)
    print(f"{'Layer':<35} {'Type':<20} {'Forward FLOPs':>12} {'Backward FLOPs':>12}")
    print("="*80)
    
    for layer in layer_list:
        fwd = layer.flops
        bwd = getattr(layer, 'backward_flops', None)
        
        if fwd is not None:
            total_forward += fwd
        if bwd is not None:
            total_backward += bwd
        
        # Skip input/output/buffer layers for cleaner output
        if layer.layer_type in ('input', 'output', 'buffer'):
            continue
            
        print(f"{layer.layer_label:<35} {layer.layer_type:<20} {format_flops(fwd):>12} {format_flops(bwd):>12}")
    
    print("="*80)
    print(f"{'TOTAL':<35} {'':<20} {format_flops(total_forward):>12} {format_flops(total_backward):>12}")
    print("="*80)
    
    return total_forward, total_backward


if __name__ == "__main__":
    # Test with VGG11
    print("\n" + "="*80)
    print("Testing VGG11")
    print("="*80)
    
    model = torchvision.models.vgg11().eval()
    x = torch.rand(1, 3, 224, 224)
    
    # Log model forward pass
    model_history = tl.log_forward_pass(model, x, vis_opt='none', save_function_args=True)
    layer_list = model_history.layer_list
    
    # Prepare fast replay
    label2idx = prepare_replay_graph(layer_list)
    
    # Execute replay
    res1 = model_log_forward_fast(layer_list, x, label2idx)
    res2 = model(x)
    
    print("Replay validation:", "PASS" if torch.allclose(res1, res2) else "FAIL")
    
    # Print FLOPs summary
    total_fwd, total_bwd = print_flops_summary(layer_list)
    
    # Test with ResNet18
    print("\n" + "="*80)
    print("Testing ResNet18")
    print("="*80)
    
    model = torchvision.models.resnet18().eval()
    model_history = tl.log_forward_pass(model, x, vis_opt='none', save_function_args=True)
    layer_list = model_history.layer_list
    label2idx = prepare_replay_graph(layer_list)
    
    res1 = model_log_forward_fast(layer_list, x, label2idx)
    res2 = model(x)
    
    print("Replay validation:", "PASS" if torch.allclose(res1, res2) else "FAIL")
    total_fwd, total_bwd = print_flops_summary(layer_list)
    
    # Test with ViT
    print("\n" + "="*80)
    print("Testing ViT-B/32")
    print("="*80)
    
    model = torchvision.models.vit_b_32(weights=None).eval()
    model_history = tl.log_forward_pass(model, x, vis_opt='none', save_function_args=True)
    layer_list = model_history.layer_list
    label2idx = prepare_replay_graph(layer_list)
    
    res1 = model_log_forward_fast(layer_list, x, label2idx)
    res2 = model(x)
    
    print("Replay validation:", "PASS" if torch.allclose(res1, res2) else "FAIL")
    total_fwd, total_bwd = print_flops_summary(layer_list)
