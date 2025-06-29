from gaussian_splatting import GaussianModel
from .noise import NoiseWrapper
from .relocate import BaseRelocationTrainer


def MCMCTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return NoiseWrapper(BaseRelocationTrainer, model, scene_extent, *args, **kwargs)
