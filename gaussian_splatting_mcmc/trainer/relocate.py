import math
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from .diff_gaussian_rasterization import compute_relocation

# https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/utils/reloc_utils.py#L5
N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)


def compute_relocation_cuda(opacity_old, scale_old, N):
    # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/utils/reloc_utils.py#L11
    N.clamp_(min=1, max=N_max-1)
    return compute_relocation(opacity_old, scale_old, N, binoms, N_max)


def replace_tensors_to_optimizer(model: GaussianModel, optimizer: torch.optim.Optimizer, inds=None):
    # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L411
    tensors_dict = {
        "xyz": model._xyz,
        "f_dc": model._features_dc,
        "f_rest": model._features_rest,
        "opacity": model._opacity,
        "scaling": model._scaling,
        "rotation": model._rotation}

    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if not group["name"] in tensors_dict:
            continue
        assert len(group["params"]) == 1
        tensor = tensors_dict[group["name"]]

        stored_state = optimizer.state.get(group['params'][0], None)
        if inds is not None:
            stored_state["exp_avg"][inds] = 0
            stored_state["exp_avg_sq"][inds] = 0
        else:
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]

    model._xyz = optimizable_tensors["xyz"]
    model._features_dc = optimizable_tensors["f_dc"]
    model._features_rest = optimizable_tensors["f_rest"]
    model._opacity = optimizable_tensors["opacity"]
    model._scaling = optimizable_tensors["scaling"]
    model._rotation = optimizable_tensors["rotation"]

    torch.cuda.empty_cache()

    return optimizable_tensors
