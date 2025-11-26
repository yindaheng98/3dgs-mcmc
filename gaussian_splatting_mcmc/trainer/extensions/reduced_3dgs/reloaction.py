from functools import partial
from typing import Callable, List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier, DensificationTrainer
from gaussian_splatting_mcmc.trainer import RelocationDensifierWrapper
from reduced_3dgs import FullReducedDensificationDensifierWrapper, FullPruningDensifierWrapper


# Combinations of Relocation and Full Pruning Densifier

def FullPruningRelocationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: List[Camera],
        *args, **kwargs) -> AbstractDensifier:
    return FullPruningDensifierWrapper(
        partial(RelocationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def FullPruningRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: List[Camera],
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullPruningRelocationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseFullPruningRelocationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return FullPruningRelocationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullPruningRelocationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseFullPruningRelocationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


FullPruningRelocationTrainer = DepthFullPruningRelocationTrainer

# Combinations of Relocation and Full Reduced Densifier


def FullReducedRelocationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: List[Camera],
        *args, **kwargs) -> AbstractDensifier:
    return FullReducedDensificationDensifierWrapper(
        partial(RelocationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def FullReducedRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: List[Camera],
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullReducedRelocationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseFullReducedRelocationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return FullReducedRelocationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullReducedRelocationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseFullReducedRelocationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


FullReducedRelocationTrainer = DepthFullReducedRelocationTrainer
