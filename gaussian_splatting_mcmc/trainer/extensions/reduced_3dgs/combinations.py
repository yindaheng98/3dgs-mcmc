from typing import List
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import CameraTrainerWrapper, NoopDensifier, DepthTrainerWrapper, OpacityResetTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs.combinations import CameraTrainableVariableSHGaussianModel
from gaussian_splatting_mcmc.trainer import MCMCTrainer
from .trainer import PrunerInMCMCTrainerWrapper
from .importance import ImportancePruner


def FullPrunerInMCMCTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=15000,
        importance_prune_until_iter=20000,
        importance_prune_interval: int = 1000,
        importance_score_resize=None,
        importance_prune_type="comprehensive",
        importance_prune_percent=0.1,
        importance_prune_thr_important_score=None,
        importance_prune_thr_v_important_score=3.0,
        importance_prune_thr_max_v_important_score=None,
        importance_prune_thr_count=1,
        importance_prune_thr_T_alpha=1.0,
        importance_prune_thr_T_alpha_avg=0.001,
        importance_v_pow=0.1,
        **kwargs):
    return PrunerInMCMCTrainerWrapper(
        lambda model, scene_extent, dataset: ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
            importance_score_resize=importance_score_resize,
            importance_prune_type=importance_prune_type,
            importance_prune_percent=importance_prune_percent,
            importance_prune_thr_important_score=importance_prune_thr_important_score,
            importance_prune_thr_v_important_score=importance_prune_thr_v_important_score,
            importance_prune_thr_max_v_important_score=importance_prune_thr_max_v_important_score,
            importance_prune_thr_count=importance_prune_thr_count,
            importance_prune_thr_T_alpha=importance_prune_thr_T_alpha,
            importance_prune_thr_T_alpha_avg=importance_prune_thr_T_alpha_avg,
            importance_v_pow=importance_v_pow,
        ),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullPrunerInMCMCTrainer(model: GaussianModel, scene_extent: float, dataset: CameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        FullPrunerInMCMCTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


def OpacityResetFullPrunerInMCMCTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        lambda model, scene_extent, *args, **kwargs: DepthFullPrunerInMCMCTrainer(model, scene_extent, dataset, *args, **kwargs),
        model, scene_extent,
        *args, **kwargs
    )


MCMCFullTrainer = OpacityResetFullPrunerInMCMCTrainer


def SHCullingMCMCTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: MCMCTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingMCMCFullTrainer(
        model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetFullPrunerInMCMCTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraMCMCFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        OpacityResetFullPrunerInMCMCTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingMCMCTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingMCMCTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingMCMCFullTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingMCMCFullTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
