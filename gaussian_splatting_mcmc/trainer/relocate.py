import math
from typing import Callable
import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractDensifier, DensifierWrapper, DensificationTrainer, DensificationInstruct
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


def _unique_reinit(n: int, reinit_idx: torch.Tensor, replace_opacity: torch.Tensor, replace_scaling: torch.Tensor):
    reinit_mask = torch.zeros(n, device=reinit_idx.device, dtype=torch.bool)
    reinit_mask[reinit_idx] = True
    tmp_opacity = torch.zeros((n, *replace_opacity.shape[1:]), device=replace_opacity.device, dtype=replace_opacity.dtype)
    tmp_opacity[reinit_idx] = replace_opacity
    tmp_scaling = torch.zeros((n, *replace_scaling.shape[1:]), device=replace_scaling.device, dtype=replace_scaling.dtype)
    tmp_scaling[reinit_idx] = replace_scaling
    return reinit_mask, tmp_opacity[reinit_mask], tmp_scaling[reinit_mask]


class Relocater(DensifierWrapper):

    def __init__(
            self, densifier: AbstractDensifier,
            cap_max=1_000_000,
            relocate_from_iter=500,
            relocate_until_iter=25_000,
            relocate_interval=100,
    ):
        super().__init__(densifier)
        self.cap_max = cap_max
        self.densify_from_iter = relocate_from_iter
        self.densify_until_iter = relocate_until_iter
        self.densify_interval = relocate_interval

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
            replace_xyz,
            replace_features_dc,
            replace_features_rest,
            replace_opacity,
            replace_scaling,
            replace_rotation,
        ) = _update_params(model, reinit_idx, ratio=ratio)

        # (
        #     model._xyz[dead_indices],
        #     model._features_dc[dead_indices],
        #     model._features_rest[dead_indices],
        #     model._opacity[dead_indices],
        #     model._scaling[dead_indices],
        #     model._rotation[dead_indices]
        # ) = _update_params(model, reinit_idx, ratio=ratio)

        replace_indices = dead_indices.argsort()
        relocation = DensificationInstruct(
            replace_xyz_mask=dead_mask,
            replace_xyz=replace_xyz[replace_indices],
            replace_features_dc_mask=dead_mask,
            replace_features_dc=replace_features_dc[replace_indices],
            replace_features_rest_mask=dead_mask,
            replace_features_rest=replace_features_rest[replace_indices],
            replace_opacity_mask=dead_mask,
            replace_opacity=replace_opacity[replace_indices],
            replace_scaling_mask=dead_mask,
            replace_scaling=replace_scaling[replace_indices],
            replace_rotation_mask=dead_mask,
            replace_rotation=replace_rotation[replace_indices],
        )

        # model._opacity[reinit_idx] = model._opacity[dead_indices]
        # model._scaling[reinit_idx] = model._scaling[dead_indices]

        reinit_mask, replace_opacity, replace_scaling = _unique_reinit(dead_mask.shape[0], reinit_idx, replace_opacity, replace_scaling)
        reinitialization = DensificationInstruct(
            replace_opacity_mask=reinit_mask,
            replace_opacity=replace_opacity,
            replace_scaling_mask=reinit_mask,
            replace_scaling=replace_scaling,
        )

        # replace_tensors_to_optimizer(model, self.optimizer, inds=reinit_idx)

        return DensificationInstruct.merge(reinitialization, relocation)

    def add_new_gs(self, cap_max) -> DensificationInstruct:
        model = self.model
        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L504
        current_num_points = model._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return DensificationInstruct()

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

        # model._opacity[add_idx] = new_opacity
        # model._scaling[add_idx] = new_scaling

        # ret = densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        # replace_tensors_to_optimizer(model, self.optimizer, inds=add_idx)

        reinit_mask, replace_opacity, replace_scaling = _unique_reinit(model._opacity.shape[0], add_idx, new_opacity, new_scaling)
        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            replace_opacity_mask=reinit_mask,
            replace_opacity=replace_opacity,
            replace_scaling_mask=reinit_mask,
            replace_scaling=replace_scaling,
        )

    def relocate_and_add_new_gs(self):
        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/train.py#L125
        dead_mask = (self.model.get_opacity <= 0.005).squeeze(-1)
        return DensificationInstruct.merge(
            self.relocate_gs(dead_mask=dead_mask),
            self.add_new_gs(cap_max=self.cap_max)
        )

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        instruct = super().densify_and_prune(loss, out, camera, step)
        if self.densify_from_iter <= step <= self.densify_until_iter and step % self.densify_interval == 0:
            instruct = DensificationInstruct.merge(instruct, self.relocate_and_add_new_gs())
        return instruct


def RelocationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        cap_max=1_000_000,
        relocate_from_iter=500,
        relocate_until_iter=25_000,
        relocate_interval=100,
        **kwargs):
    return Relocater(
        base_densifier_constructor(model, scene_extent, *args, **kwargs),
        cap_max=cap_max,
        relocate_from_iter=relocate_from_iter,
        relocate_until_iter=relocate_until_iter,
        relocate_interval=relocate_interval,
    )


def RelocationDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        cap_max=1_000_000,
        relocate_from_iter=500,
        relocate_until_iter=25_000,
        relocate_interval=100,
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = Relocater(
        densifier,
        cap_max=cap_max,
        relocate_from_iter=relocate_from_iter,
        relocate_until_iter=relocate_until_iter,
        relocate_interval=relocate_interval,
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )

# similar to gaussian_splatting.trainer.densifier.densifier
