from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs.importance import ImportancePruningDensifierWrapper


# Combinations of Relocation and Full Reduced Densifier

def MCMCImportancePruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return MCMCTrainerWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseMCMCImportancePruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return MCMCImportancePruningTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthMCMCImportancePruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseMCMCImportancePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


MCMCImportancePruningTrainer = DepthMCMCImportancePruningTrainer


# Full Pruning Trainer + SH Culling

def SHCullingMCMCImportancePruningTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        MCMCImportancePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
