import contextlib
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def set_seeds_all(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def kl_logit(logit_pred, logit_ori):
    prob_pred = F.softmax(logit_pred, dim=1)
    prob_ori = F.softmax(logit_ori, dim=1)
    # kl_loss = torch.sum(prob_pred * (torch.log(prob_pred) - torch.log(prob_ori)), dim=1)
    kl_loss = torch.sum((prob_pred - prob_ori)**2, dim=1)
    return torch.mean(kl_loss)

def ada_kl_logit(logit_pred, logit_ori, gamma):
    prob_pred = F.softmax(logit_pred, dim=1)
    prob_ori = F.softmax(logit_ori, dim=1)
    kl_loss = torch.sum(prob_pred * (torch.log(prob_pred) - torch.log(prob_ori)), dim=1)
    kl_loss =  torch.pow(kl_loss, gamma+1)
    return torch.mean(kl_loss)

def dict_mean_std(dict_list):
    keys_all = dict_list[0].keys()
    results = {}
    for key_single in keys_all:
        value_list = [dict_list[i][key_single] for i in range(len(dict_list))]
        # print(f'value_list={type(value_list[0])}')
        # print(f'value_list={value_list}')
        if isinstance(value_list[0], (bool, dict)):
            results[key_single] = value_list
        else:
            value_mean = round(np.mean(value_list), 2)
            value_std = round(np.std(value_list), 2)
            results[key_single] = [value_mean, value_std]
    
    assert results.keys() == keys_all
    return results




@contextlib.contextmanager
def training_mode(*modules, is_train:bool):
    group = nn.ModuleList(modules)
    was_training = {module: module.training for module in group.modules()}
    try:
        yield group.train(is_train)
    finally:
        for key, module in group.named_modules():
            if module in was_training:
                module.training = was_training[module]
            else:
                raise ValueError("Model was modified inside training_mode(...) context, could not find {}".format(key))


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large data tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(data), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def safe_backward(loss, parameters, accumulate=1, allow_unused=False):
    parameters = list(parameters)  # Capture the generator output
    grads = torch.autograd.grad(loss, parameters, allow_unused=allow_unused)
    nan, inf = False, False
    for g in grads:
        if g is not None:
            nan |= g.isnan().any().item()
            inf |= g.isinf().any().item()

    if not (nan or inf):
        for p, g in zip(parameters, grads):
            if g is None:
                continue

            if p.grad is None:
                p.grad = g / accumulate
            else:
                p.grad += g / accumulate