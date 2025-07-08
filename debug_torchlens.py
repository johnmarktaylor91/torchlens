import torch
import torchvision
import torchlens as tl

def label2index(net_list):
    """Map layer_label to index."""
    return {layer.layer_label: i for i, layer in enumerate(net_list)}

def model_log_forward_eval(net_list, label2index, x):
    """Layer-by-layer forward computation in eval mode."""
    for layer in net_list:
        if layer.func_applied_name == 'none':
            if layer.layer_type != 'buffer':
                layer.tensor_contents = None
            continue

        x_in, buffer_in = [], []
        for parent_label in layer.parent_layers:
            parent_idx = label2index[parent_label]
            parent = net_list[parent_idx]
            if parent.layer_type != 'buffer':
                # If parent tensor is None and operation number matches, use current input
                if parent.tensor_contents is None and parent.operation_num == layer.operation_num - 1:
                    x_in.append(x)
                else:
                    x_in.append(parent.tensor_contents)
                    # If the last child of the parent has operation_num < current+1 and parent is not buffer, release parent tensor
                    last_child_label = parent.child_layers[-1]
                    last_child = net_list[label2index[last_child_label]]
                    if last_child.operation_num < layer.operation_num + 1 and parent.layer_type != 'buffer':
                        parent.tensor_contents = None
            else:
                buffer_in.append(parent.tensor_contents)

        # Forward computation
        x = layer.func_applied(*x_in, *(layer.parent_params), *buffer_in, *(layer.func_all_args_non_tensor))
        # Decide whether to release current layer tensor
        if layer.child_layers and net_list[label2index[layer.child_layers[-1]]].operation_num <= layer.operation_num + 1:
            layer.tensor_contents = None
        else:
            layer.tensor_contents = x
    return x

def model_log_forward_train(net_list, label2index, x):
    """Layer-by-layer forward computation in train mode (keep gradients)."""
    first_in = True
    for layer in net_list:
        if layer.func_applied_name == 'none':
            if layer.layer_type == 'input':
                layer.tensor_contents = x
            elif layer.layer_type == 'output':
                parent = net_list[label2index[layer.parent_layers[0]]]
                layer.tensor_contents = parent.tensor_contents
            continue

        x_in, buffer_in = [], []
        for parent_label in layer.parent_layers:
            if parent_label not in label2index:
                if first_in:
                    x_in.append(x.clone())
                    first_in = False
            else:
                parent = net_list[label2index[parent_label]]
                if parent.layer_type != 'buffer':
                    x_in.append(parent.tensor_contents.clone())
                else:
                    buffer_in.append(parent.tensor_contents)
        layer.tensor_contents = layer.func_applied(*x_in, *(layer.parent_params), *buffer_in, *(layer.func_all_args_non_tensor))
    return net_list[-1].tensor_contents

if __name__ == "__main__":
    model = torchvision.models.vgg11().eval()
    x = torch.rand(1, 3, 224, 224)
    model_history = tl.log_forward_pass(model, x, vis_opt='none', save_function_args=True)
    layer_list = model_history.layer_list
    label2index_dict = label2index(layer_list)
    res1 = model_log_forward_eval(layer_list, label2index_dict, x)
    res2 = model(x)
    print("The outputs are the same." if torch.allclose(res1, res2) else "The outputs are different.")