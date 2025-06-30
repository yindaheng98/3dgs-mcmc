from typing import List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCDensifierTrainerWrapper
from reduced_3dgs.importance import ImportancePruner

# Importance Pruner


def ImportancePrunerInMCMCTrainer(
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
    return MCMCDensifierTrainerWrapper(
        lambda model, scene_extent: ImportancePruner(
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
        model,
        scene_extent,
        *args, **kwargs
    )


# Depth trainer
def DepthImportancePrunerInMCMCTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        ImportancePrunerInMCMCTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


MCMCImportanceTrainer = DepthImportancePrunerInMCMCTrainer
BaseMCMCImportanceTrainer = ImportancePrunerInMCMCTrainer
