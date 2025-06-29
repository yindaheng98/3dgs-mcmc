import math
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, DensificationTrainer
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


def _update_params(model: GaussianModel, idxs, ratio):
    # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L451
    new_opacity, new_scaling = compute_relocation_cuda(
        opacity_old=model.get_opacity[idxs, 0],
        scale_old=model.get_scaling[idxs],
        N=ratio[idxs, 0] + 1
    )
    new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
    new_opacity = model.inverse_opacity_activation(new_opacity)
    new_scaling = model.scaling_inverse_activation(new_scaling.reshape(-1, 3))
    return model._xyz[idxs], model._features_dc[idxs], model._features_rest[idxs], new_opacity, new_scaling, model._rotation[idxs]


def _sample_alives(probs, num, alive_indices=None):
    # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L464
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    sampled_idxs = torch.multinomial(probs, num, replacement=True)
    if alive_indices is not None:
        sampled_idxs = alive_indices[sampled_idxs]
    ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
    return sampled_idxs, ratio


class Relocater(DensificationTrainer):

    def relocate_gs(self, dead_mask=None):
        model = self.model
        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L474
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (model.get_opacity[alive_indices, 0])
        reinit_idx, ratio = _sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            model._xyz[dead_indices],
            model._features_dc[dead_indices],
            model._features_rest[dead_indices],
            model._opacity[dead_indices],
            model._scaling[dead_indices],
            model._rotation[dead_indices]
        ) = _update_params(model, reinit_idx, ratio=ratio)

        model._opacity[reinit_idx] = model._opacity[dead_indices]
        model._scaling[reinit_idx] = model._scaling[dead_indices]

        replace_tensors_to_optimizer(model, self.optimizer, inds=reinit_idx)

    def add_new_gs(self, cap_max, densification_postfix):
        model = self.model
        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L504
        current_num_points = model._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = model.get_opacity.squeeze(-1)
        add_idx, ratio = _sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation
        ) = _update_params(model, add_idx, ratio=ratio)

        model._opacity[add_idx] = new_opacity
        model._scaling[add_idx] = new_scaling

        densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        replace_tensors_to_optimizer(model, self.optimizer, inds=add_idx)

        return num_gs

    def relocate_and_densify(self):
        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/train.py#L125
        dead_mask = (self.model.get_opacity <= 0.005).squeeze(-1)
        self.relocate_gs(dead_mask=dead_mask)
        self.add_new_gs(cap_max=self.cap_max)
