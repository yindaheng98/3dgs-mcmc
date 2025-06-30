from gaussian_splatting import GaussianModel
from .noise import NoiseWrapper
from .relocate import BaseRelocationTrainer
from .scale_opacity_reg import ScaleOpacityRegularizeTrainerWrapper


def NoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return NoiseWrapper(BaseRelocationTrainer, model, scene_extent, *args, **kwargs)


def MCMCTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(NoiseRelocationTrainer, model, scene_extent, *args, **kwargs)
