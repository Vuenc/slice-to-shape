import yaml
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from torch.types import Number

from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter
from ssm_thyroid_dataset import SSMvoxelDataset
from encoder import SdfVoxelMatchNet
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, dataset
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
import os
from datetime import datetime

import sys
sys.path.append("../")
from block_batch_sampler import BlockRandomBatchSampler
from soft_assignment_slice_matching import soft_assignment_matrix, soft_assignment_loss
import soft_assignment_slice_matching

def train(
    train_dataset: SSMvoxelDataset,
    val_dataset: Optional[SSMvoxelDataset],
    optimizer_params: Dict[Any, Any] = {"lr": 1e-3},
    embedding_size=128,
    batch_size=50,
    epochs=50,
    epoch_iteration_offsets: Tuple[int, int] = (0,0),
    tensorboard=True,
    model: Optional[SdfVoxelMatchNet]=None,
    normalize_embeddings=True,
    validate_every_n_epochs=1,
    save_every_n_epochs: Optional[int]=5,
    save_dir="saves/",
    log_dir=None,
    device: str="cuda:1",
    ultrasound_patches_shape=None,
    sdf_patches_shape=None,
    reshuffle_samples=True,
    # positive_sample_perturb: float=0.0,
    loss_type: Union[Literal["weightedSoftMarginTriplet"], Literal["classicalTriplet"], Literal["geometricDistanceL2"],
        Literal["softAssigment"]] = "weightedSoftMarginTriplet",
    loss_params: Dict[Any, Any] = {"alpha": 5.0},
    # negative_sample_threshold: float=40.0,
    data_parallel_gpu_ids: Optional[List[int]] = None,
    negative_sample_from_p_furthest = None,
    dot_product_similarity: bool=False,
    stopping_patience: Optional[int]=0,
    num_workers: int=0,
    pin_memory: bool=False,
):
    last_best_model_path = None
    stopping_counter = 0

    if ultrasound_patches_shape is None:
        ultrasound_patches_shape = train_dataset.ultrasound_patches_shape
    if sdf_patches_shape is None:
        sdf_patches_shape = train_dataset.sdf_patches_shape
    if negative_sample_from_p_furthest is None:
        negative_sample_from_p_furthest = train_dataset.negative_sample_from_p_furthest

    if model is None:
        model = SdfVoxelMatchNet(embedding_size=embedding_size, normalize_embeddings=normalize_embeddings,
            ultrasound_patches_shape=ultrasound_patches_shape, sdf_patches_shape=sdf_patches_shape).to(device)
    assert(model is not None)
    
    if loss_type == "softAssignment":
        # Add a trainable parameter for the "sink value"
        model.add_sink_value_parameter(torch.tensor([-2.1]).to(device))
        model.to(device)
        assert("exponentBeta" in loss_params)
    if data_parallel_gpu_ids is not None:
        model = torch.nn.DataParallel(model, device_ids=data_parallel_gpu_ids, output_device=device) # type: ignore

    
    assert(model is not None)
    optimizer = Adam(model.parameters(), **optimizer_params)

    # train_dataset.set_positive_sample_perturb(positive_sample_perturb)
    # if val_dataset is not None:
    #     val_dataset.set_positive_sample_perturb(positive_sample_perturb)

    # if reshuffle_negative_samples or cross_thyroid:
    #     for dataset in [train_dataset] + ([val_dataset] if val_dataset is not None else []):
    #         if not cross_thyroid:
    #             dataset.reshuffle_negative_samples()
    #             dataset.fix_sdf_patches_neg(negative_sample_threshold)
    #         else:
    #             dataset.cross_shuffle(negative_sample_threshold)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=BlockRandomBatchSampler(train_dataset.get_block_sizes(),
                                              batch_size, allow_block_overlap=False),
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_sampler=BlockRandomBatchSampler(val_dataset.get_block_sizes(), batch_size, allow_block_overlap=False),
            num_workers=num_workers, pin_memory=pin_memory
        )

    statistics_keys = ["%s/%s" % (key, train_or_val) for train_or_val in ["train", "validation"]
        for key in ["loss", "distance+", "distance-", "distanceΔ", "US norm", "SDF norm+", "SDF norm-",
            "distances+", "distances-", "distancesΔ", "unmatchedPatchesRatio", "averageAssignmentScore", "assignmentMatrix"]] + ["sinkValue"]
    histogram_tags = ["%s/%s" % (key, train_or_val) for train_or_val in ["train", "validation"]
        for key in ["distances+", "distances-", "distancesΔ"]]
    image_tags = ["%s/%s" % (key, train_or_val) for train_or_val in ["train", "validation"]
        for key in ["assignmentMatrix"]]
    epoch_offset, iteration_offset = epoch_iteration_offsets
    
    hyperparams = [("lr", optimizer_params["lr"]),
        ("batchSize", batch_size),
        ("epochs", epochs if epoch_offset == 0 else ("%d-%d" % (epoch_offset, epoch_offset+epochs))),
        ("embeddingSize", embedding_size),
        ("usPatch", "x".join(map(str, ultrasound_patches_shape))),
        ("sdfPatch", "x".join(map(str, sdf_patches_shape))),
        (None, "normalized" if normalize_embeddings else "unnormalized"),
        (None, ("reshuffle" if reshuffle_samples else "noReshuffle")),
        ("pFurthest", negative_sample_from_p_furthest if loss_type != "softAssignment" else None),
        # ("posPerturb", positive_sample_perturb),
        ("loss", "%s(%s)" % (loss_type, ",".join("%s=%s" % (param, value) for param, value in loss_params.items()))),
        # ("negSampleThreshold", negative_sample_threshold),
        (None, ("embDot" if dot_product_similarity else "embDist")), # if loss_type != "softAssignment" else None),
    ]
        
    log_comment = "_"
    writer = SummaryWriter(log_dir, log_comment) if tensorboard else None
    # for saving the model, (somewhat) mirroring the format the tensorboard summary write uses (but without the hostname)
    time_string = datetime.now().strftime("%b%d_%H-%M-%S")
    
    if save_every_n_epochs is not None:
        os.makedirs(save_dir, exist_ok=True)
        # save the hyperparameters
        yaml_file = Path(save_dir) / f'model_hyperparams_{time_string}.yml'
        with yaml_file.open('w') as f:
            yaml.dump(hyperparams, f)

    train_dataset.set_patches_shapes_and_p_furthest(ultrasound_patches_shape, sdf_patches_shape, negative_sample_from_p_furthest)
    if val_dataset is not None:
        val_dataset.set_patches_shapes_and_p_furthest(ultrasound_patches_shape, sdf_patches_shape, negative_sample_from_p_furthest)

    total_iteration = iteration_offset
    best_loss = np.inf
    
    for epoch in tqdm(range(epoch_offset, epoch_offset + epochs)):
        model.train()        
        statistics: Dict[str, List[Union[Number, np.ndarray]]] = dict([(key, []) for key in statistics_keys])

        for i, (us_patch, sdf_patch_pos, sdf_patch_neg, us_coords, sdf_pos_coords, sdf_neg_coords) in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            run_one_iteration(model, us_patch, sdf_patch_pos, sdf_patch_neg, us_coords, sdf_pos_coords, sdf_neg_coords,
                statistics, loss_type, loss_params, dot_product_similarity, device, train=True, optimizer=optimizer)
            if tensorboard:
                write_to_tensorboard(statistics, writer, iteration=total_iteration, histogram_tags=histogram_tags, image_tags=image_tags)
            total_iteration += 1
        
        if tensorboard:
            write_to_tensorboard(statistics, writer, epoch=epoch, histogram_tags=histogram_tags, image_tags=image_tags)

        if val_dataloader is not None and epoch % validate_every_n_epochs == validate_every_n_epochs - 1:
            model.eval()
            statistics = dict([(key, []) for key in statistics_keys])

            for i, (us_patch, sdf_patch_pos, sdf_patch_neg, us_coords, sdf_pos_coords, sdf_neg_coords) in enumerate(val_dataloader):
                run_one_iteration(model, us_patch, sdf_patch_pos, sdf_patch_neg, us_coords, sdf_pos_coords, sdf_neg_coords,
                    statistics, loss_type, loss_params, dot_product_similarity, device, train=False)

            if tensorboard:
                write_to_tensorboard(statistics, writer, epoch=epoch, histogram_tags=histogram_tags, image_tags=image_tags)

        val_loss=np.mean(statistics["loss/validation"])
        if save_every_n_epochs is not None and epoch % save_every_n_epochs == 0:
            _model_path = Path(save_dir) / f"model_{time_string}_epoch-{epoch}_val_loss-{val_loss}.obj"
            torch.save(model, _model_path)
        if val_loss<best_loss:
            if last_best_model_path is not None and last_best_model_path.exists():
                os.remove(last_best_model_path)
            last_best_model_path = Path(save_dir) / f"best_model_{time_string}_epoch-{epoch}_val_loss-{val_loss}.obj"
            torch.save(model, last_best_model_path)
            best_loss=val_loss
            stopping_counter=0
        else:
            stopping_counter+=1
            
        if stopping_patience>0:
            if stopping_patience==stopping_counter:
                break
        
        
        if reshuffle_samples:
            train_dataset.reshuffle_samples()
#         for dataset in [train_dataset] + ([val_dataset] if val_dataset is not None else []):
#             if not cross_thyroid:
#                 dataset.reshuffle_negative_samples()
#                 dataset.fix_sdf_patches_neg(negative_sample_threshold)
#             else:
#                 dataset.cross_shuffle(negative_sample_threshold)
    return model
        

def run_one_iteration(model, us_patch, sdf_patch_pos, sdf_patch_neg,
        us_coords, sdf_pos_coords, sdf_neg_coords, # the coords represent the centers of their respective patches
        statistics: Dict[str, List[Union[Number, np.ndarray]]], loss_type: str, loss_params: Dict[Any, Any],
        dot_product_similarity: bool, device: str, train: bool, optimizer: Optional[Any]=None):
    # Move to device (i.e. GPU, if available)
    us_patch, sdf_patch_pos, sdf_patch_neg = us_patch.to(device), sdf_patch_pos.to(device), sdf_patch_neg.to(device)
    if train:
        assert(optimizer is not None)
        optimizer.zero_grad()

    embeddings_us, embeddings_sdf = model.forward(us_patch, torch.cat([sdf_patch_pos, sdf_patch_neg]))
    embeddings_sdf_pos, embeddings_sdf_neg = torch.split(embeddings_sdf, sdf_patch_pos.shape[0])

    dists_pos = torch.norm(embeddings_us - embeddings_sdf_pos, dim=1)
    dists_neg = torch.norm(embeddings_us - embeddings_sdf_neg, dim=1)

    sink_value = unmatched_patches_ratio = average_assignment_score = assignment_matrix = None

    if loss_type == "weightedSoftMarginTriplet":
        loss = torch.log(1 + torch.exp(loss_params["alpha"] * (dists_pos - dists_neg))).mean()
    elif loss_type == "classicalTriplet":
        loss = torch.maximum(dists_pos - dists_neg + loss_params["alpha"], torch.zeros_like(dists_pos)).mean()
    elif loss_type == "geometricDistanceL2":
        # Move geometric coordinates to device (i.e. GPU, if available)
        us_coords, sdf_pos_coords, sdf_neg_coords = us_coords.to(device), sdf_pos_coords.to(device), sdf_neg_coords.to(device)
        geometric_distances_pos = torch.norm(us_coords - sdf_pos_coords, dim=1)
        geometric_distances_neg = torch.norm(us_coords - sdf_neg_coords, dim=1)
        loss = ((geometric_distances_pos - dists_pos) ** 2).mean() + ((geometric_distances_neg - dists_neg) ** 2).mean()
    elif loss_type == "softAssignment":
        sink_value = model.module.sink_value if isinstance(model, torch.nn.DataParallel) else model.sink_value
        us_coords, sdf_pos_coords, sdf_neg_coords = us_coords.to(device), sdf_pos_coords.to(device), sdf_neg_coords.to(device)
        similarity_function = (soft_assignment_slice_matching.dot_product_similarity if dot_product_similarity
            else soft_assignment_slice_matching.negative_distance_similarity)
        assignment_matrix = soft_assignment_matrix(embeddings_us, torch.vstack([embeddings_sdf_pos, embeddings_sdf_neg]), sink_value,
            similarity_function=similarity_function)
        loss = soft_assignment_loss(assignment_matrix, us_coords, torch.vstack([sdf_pos_coords, sdf_neg_coords]), loss_params["exponentBeta"])
        unmatched_patches_ratio = (assignment_matrix.argmax(dim=1) == assignment_matrix.shape[1] - 1).float().mean()
        average_assignment_score = assignment_matrix[:-1,:-1].mean()
    else:
        raise ValueError("`loss_type` must be one of: ['weightedSoftMarginTriplet', 'classicalTriplet', 'geometricDistanceL2', 'softAssignment']")
    if train:
        loss.backward()
        assert(optimizer is not None)
        optimizer.step()

    update_statistics(statistics, "train" if train else "validation", loss, dists_pos, dists_neg,
        embeddings_us, embeddings_sdf_pos, embeddings_sdf_neg, sink_value, unmatched_patches_ratio, average_assignment_score, assignment_matrix)

def update_statistics(statistics: Dict[str, List[Union[Number, np.ndarray]]], train_or_val: Union[Literal["train"], Literal["validation"]],
        loss, dists_pos, dists_neg, embeddings_us, embeddings_sdf_pos, embeddings_sdf_neg, sink_value=None, unmatched_patches_ratio=None,
        average_assignment_score=None, assignment_matrix=None):
    statistics["loss/" + train_or_val].append(loss.item())
    statistics["distance+/" + train_or_val].append(dists_pos.mean().item())
    statistics["distance-/" + train_or_val].append(dists_neg.mean().item())
    statistics["distanceΔ/" + train_or_val].append((dists_pos - dists_neg).mean().item())
    statistics["US norm/" + train_or_val].append(torch.norm(embeddings_us, dim=1).mean().item())
    statistics["SDF norm+/" + train_or_val].append(torch.norm(embeddings_sdf_pos, dim=1).mean().item())
    statistics["SDF norm-/" + train_or_val].append(torch.norm(embeddings_sdf_neg, dim=1).mean().item())
    statistics["distances+/" + train_or_val].append(dists_pos.detach().cpu().numpy())
    statistics["distances-/" + train_or_val].append(dists_neg.detach().cpu().numpy())
    statistics["distancesΔ/" + train_or_val].append((dists_pos - dists_neg).detach().cpu().numpy())
    if sink_value is not None: statistics["sinkValue"].append(sink_value.item())
    if unmatched_patches_ratio is not None: statistics["unmatchedPatchesRatio/" + train_or_val].append(unmatched_patches_ratio.item())
    if average_assignment_score is not None: statistics["averageAssignmentScore/" + train_or_val].append(average_assignment_score.item())
    if assignment_matrix is not None: statistics["assignmentMatrix/" + train_or_val].append(assignment_matrix.detach().cpu().numpy())

def write_to_tensorboard(statistics: Dict[str, List[Union[Number, np.ndarray]]], writer, epoch: Optional[int]=None,
        iteration: Optional[int]=None, histogram_tags: List[str]=[], image_tags: List[str]=[]):
    assert((epoch is not None) ^ (iteration is not None))
    for tag, values in statistics.items():
        if len(values) > 0:
            if tag not in histogram_tags and tag not in image_tags:
                # in this case `values` should be a List[Number]
                if epoch is not None:
                    writer.add_scalar(tag + "/by_epoch", np.mean(values), epoch)
                else:
                    writer.add_scalar(tag + "/by_iteration", values[-1], iteration)
            elif tag in histogram_tags:
                # in this case `values` should be a List[np.ndarray]
                if epoch is not None: 
                    if not np.isnan(np.concatenate(values)).any():
                        writer.add_histogram(tag + "/by_epoch", np.concatenate(values), epoch)
                else:
                    if not np.isnan(values[-1]).any():
                        writer.add_histogram(tag + "/by_iteration", values[-1], iteration)
            elif tag in image_tags:
                continue
                # in this case `values` should be a List[np.ndarray]
                if epoch is not None:
                    writer.add_image(tag + "/by_epoch", np.mean(values, axis=0), epoch, dataformats="HW")
                else:
                    writer.add_image(tag + "/by_iteration", values[-1], iteration, dataformats="HW")
