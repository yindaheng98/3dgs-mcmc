from functools import partial
from typing import Callable, List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs import FullReducedDensificationDensifierWrapper


# Combinations of Relocation and Full Reduced Densifier


def MCMCFullReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: List[Camera],
        *args, **kwargs):
    return MCMCTrainerWrapper(
        partial(FullReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseMCMCFullReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return MCMCFullReducedDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthMCMCFullReducedDensificationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseMCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


MCMCFullReducedDensificationTrainer = DepthMCMCFullReducedDensificationTrainer
