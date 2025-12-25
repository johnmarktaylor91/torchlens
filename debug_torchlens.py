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


if __name__ == "__main__":
    model = torchvision.models.vgg11().eval()
    x = torch.rand(1, 3, 224, 224)
    
    # 记录模型前向传播
    model_history = tl.log_forward_pass(model, x, vis_opt='unrolled', save_function_args=True)
    layer_list = model_history.layer_list
    
    # 准备快速 replay
    label2idx = prepare_replay_graph(layer_list)
    
    # 执行 replay
    res1 = model_log_forward_fast(layer_list, x, label2idx)
    res2 = model(x)
    
    print("The outputs are the same." if torch.allclose(res1, res2) else "The outputs are different.")
    
    # Print per-layer FLOPs
    print("\nLayer-wise FLOPs (forward, backward):")
    for layer in layer_list:
        print(f"{layer.layer_label:40s} | {str(layer.flops):>12} | {str(getattr(layer, 'backward_flops', None)):>12}")
