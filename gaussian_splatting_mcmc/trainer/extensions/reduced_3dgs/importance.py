from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, CameraTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, CameraTrainableVariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs.importance import ImportancePruningDensifierWrapper


# Combinations of MCMC and Importance Pruning Densifier

def MCMCImportancePruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return MCMCTrainerWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )


def BaseMCMCImportancePruningTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return MCMCImportancePruningTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthMCMCImportancePruningTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseMCMCImportancePruningTrainer,
        model, dataset,
        **configs
    )


MCMCImportancePruningTrainer = DepthMCMCImportancePruningTrainer


def CameraMCMCImportancePruningTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        MCMCImportancePruningTrainer,
        model, dataset,
        **configs
    )


# Importance Pruning Trainer + SH Culling

def SHCullingMCMCImportancePruningTrainer(
        model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        MCMCImportancePruningTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingMCMCImportancePruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingMCMCImportancePruningTrainer,
        model, dataset,
        **configs
    )
