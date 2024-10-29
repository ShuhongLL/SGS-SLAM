import argparse
import os
import random
import sys
import shutil
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_seq_params, save_params, save_params_ckpt, save_seq_params_ckpt
from utils.recon_helpers import setup_camera
from utils.gs_helpers import (
    params2rendervar, params2depthplussilhouette, semantics2rendervar,
    transformed_params2depthplussilhouette,
    transformed_semantics2rendervar,
    transform_to_frame, report_progress, eval,
    l1_loss_v1, matrix_to_quaternion
)
from utils.gs_external import (
    calc_ssim, build_rotation, densify,
    get_expon_lr_func, update_learning_rate
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def initialize_optimizer(params, lrs_dict, params_opt_exclude):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() \
                    if k not in params_opt_exclude and k in lrs]

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep_from_ckpt(ckpt_path, dataset, num_frames, lrs_dict, mean_sq_dist_method,
                                        load_semantics=False, device="cuda"):
    # Params dont need gradient
    params_opt_exclude = set()
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[0]
        semantic_id = color.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        params_opt_exclude.add("semantic_ids")
    else:
        color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)

    # Initialize Parameters & Optimizer from Checkpoint
    # Load checkpoint
    print(f"Loading Params")
    params = dict(np.load(ckpt_path, allow_pickle=True))
    variables = {}

    for k in ['intrinsics', 'w2c', 'org_width', 'org_height', 'gt_w2c_all_frames', 'keyframe_time_indices']:
    # for k in ['timestep','intrinsics', 'w2c', 'org_width', 'org_height', 'gt_w2c_all_frames']:
        params.pop(k)

    print(params.keys())
    for k in params.keys():
        if k in params_opt_exclude:
            params[k] = torch.tensor(params[k]).cuda()
        else:
            params[k] = torch.tensor(params[k]).cuda().float().requires_grad_(True)

    if load_semantics and 'semantic_ids' in params.keys():
        if params['semantic_ids'].dim() == 1:
            params['semantic_ids'] = params['semantic_ids'].unsqueeze(1)
    
    variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    # variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['timestep'] = torch.tensor(params['timestep']).cuda().float()
    params.pop('timestep')
    optimizer = initialize_optimizer(params, lrs_dict, params_opt_exclude)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/2.0

    return params, variables, optimizer, params_opt_exclude, intrinsics, w2c, cam


def get_loss_gs(params, curr_data, variables, loss_weights, load_semantics=False):
    # Initialize Loss Dictionary
    losses = {}

    # Initialize Render Variables
    rendervar = params2rendervar(params)
    depth_sil_rendervar = params2depthplussilhouette(params, curr_data['w2c'])

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]

    # Semantic Color Rendering
    if load_semantics:
        semantic_rendervar = semantics2rendervar(params)
        seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
        # Semantic color loss
        losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['semantic_color']) + \
            0.2 * (1.0 - calc_ssim(seg, curr_data['semantic_color']))

    # Get invalid Depth Mask
    valid_depth_mask = (curr_data['depth'] != 0.0)
    depth = depth * valid_depth_mask

    # RGB Loss
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    
    # Depth Loss
    losses['depth'] = l1_loss_v1(depth, curr_data['depth'])

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    return params


def infill_depth(depth, inpaint_radius=1):
    """
    Function to infill Depth for invalid regions
    Input:
        depth: Depth Image (Numpy)
        radius: Radius of the circular neighborhood for infilling
    Output:
        depth: Depth Image with invalid regions infilled (Numpy)
    """
    invalid_mask = (depth == 0)
    invalid_mask = invalid_mask.astype(np.uint8)
    filled_depth = cv2.inpaint(depth, invalid_mask, inpaint_radius, cv2.INPAINT_NS)

    return filled_depth


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    print(f"{config}")

    # Init WandB
    if config['use_wandb']:
        wandb_step = 0
        wandb_time_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
        wandb_run.define_metric("Mapping_Iters")
        wandb_run.define_metric("Number of Gaussians - Densification", step_metric="Mapping_Iters")
        wandb_run.define_metric("Learning Rate - Means3D", step_metric="Mapping_Iters")

    # Get Device
    device = torch.device(config["primary_device"])
    if config["primary_device"].startswith("cuda:"):
        device_id = int(config["primary_device"].split(':')[1])
        torch.cuda.set_device(device_id)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "load_semantics" not in dataset_config:
        load_semantics = False
        num_semantic_classes = 0
    else:
        load_semantics = dataset_config["load_semantics"]
        num_semantic_classes = dataset_config["num_semantic_classes"]
        semantic_color_all_frames_map = []
        semantic_id_all_frames_map = []

    # Poses are relative to the first frame
    mapping_dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_semantics=load_semantics,
        num_semantic_classes=num_semantic_classes,
    )

    eval_dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["eval_stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_semantics=load_semantics,
        num_semantic_classes=num_semantic_classes,
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(mapping_dataset)
    eval_num_frames = dataset_config["eval_num_frames"]
    if eval_num_frames == -1:
        eval_num_frames = len(eval_dataset)

    # Initialize Parameters, Optimizer & Canoncial Camera parameters
    ckpt_path = config["data"]["param_ckpt_path"]
    params, variables, optimizer, params_opt_exclude, intrinsics, \
        w2c, cam = initialize_first_timestep_from_ckpt(ckpt_path, mapping_dataset, num_frames,
                                                       config['train']['lrs_mapping'],
                                                       config['mean_sq_dist_method'],
                                                       load_semantics=load_semantics,
                                                       device=device)

    # Use the first frame coordinate
    if load_semantics:
        _, _, map_intrinsics, _, _, _ = mapping_dataset[0]
    else:
        _, _, map_intrinsics, _ = mapping_dataset[0]

    # Load all RGBD frames - Mapping dataloader
    color_all_frames_map = []
    depth_all_frames_map = []
    gt_w2c_all_frames_map = []
    gs_cams_all_frames_map = []
    for time_idx in range(num_frames):
        if load_semantics:
            color, depth, _, gt_pose, semantic_id, semantic_color = mapping_dataset[time_idx]
            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            semantic_id_all_frames_map.append(semantic_id)
            semantic_color_all_frames_map.append(semantic_color)
        else:
            color, depth, _, gt_pose = mapping_dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        color_all_frames_map.append(color)
        depth_all_frames_map.append(depth)
        gt_w2c_all_frames_map.append(gt_w2c)
        # Setup Gaussian Splatting Camera
        gs_cam = setup_camera(color.shape[2], color.shape[1], 
                              map_intrinsics.cpu().numpy(), 
                              gt_w2c.detach().cpu().numpy(),
                              device=device)
        gs_cams_all_frames_map.append(gs_cam)

    # Iterate over Scan
    for time_idx in tqdm(range(num_frames)):
        # Optimization Iterations
        num_iters_mapping = config['train']['num_iters_mapping']

        # Initialize current frame data
        iter_time_idx = time_idx
        color = color_all_frames_map[iter_time_idx]
        depth = depth_all_frames_map[iter_time_idx]
        curr_gt_w2c = gt_w2c_all_frames_map[:iter_time_idx+1]
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 
                     'intrinsics': intrinsics, 'w2c': w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        if load_semantics:
            semantic_id = semantic_id_all_frames_map[iter_time_idx]
            semantic_color = semantic_color_all_frames_map[iter_time_idx]
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color

        # Add new Gaussians to the scene based on the Silhouette
        # if time_idx > 0:
        #     params, variables = add_new_gaussians(params, variables, curr_data, 
        #                                           config['train']['sil_thres'], time_idx,
        #                                           config['mean_sq_dist_method'])
        post_num_pts = params['means3D'].shape[0]
        if config['use_wandb']:
            wandb_run.log({"Init/Number of Gaussians": post_num_pts,
                           "Init/step": wandb_time_step})

        # Reset Optimizer & Learning Rates for Full Map Optimization
        optimizer = initialize_optimizer(params, config['train']['lrs_mapping'], params_opt_exclude)
        means3D_scheduler = get_expon_lr_func(lr_init=config['train']['lrs_mapping']['means3D'], 
                                              lr_final=config['train']['lrs_mapping_means3D_final'],
                                              lr_delay_mult=config['train']['lr_delay_mult'],
                                              max_steps=config['train']['num_iters_mapping'])
        
        # Mapping
        if (time_idx + 1) == num_frames:
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                # Update Learning Rates for means3D
                updated_lr = update_learning_rate(optimizer, means3D_scheduler, iter+1)
                if config['use_wandb']:
                    wandb_run.log({"Learning Rate - Means3D": updated_lr})
                # Randomly select a frame until current time step
                iter_time_idx = random.randint(0, time_idx)
                # Initialize Data for selected frame
                iter_color = color_all_frames_map[iter_time_idx]
                iter_depth = depth_all_frames_map[iter_time_idx]
                iter_gt_w2c = gt_w2c_all_frames_map[:iter_time_idx+1]
                iter_gs_cam = gs_cams_all_frames_map[iter_time_idx]
                iter_data = {'cam': iter_gs_cam, 'im': iter_color, 'depth': iter_depth, 
                             'id': iter_time_idx, 'intrinsics': map_intrinsics, 
                             'w2c': gt_w2c_all_frames_map[iter_time_idx], 'iter_gt_w2c_list': iter_gt_w2c}
                if load_semantics:
                    iter_data['semantic_id'] = semantic_id_all_frames_map[iter_time_idx]
                    iter_data['semantic_color'] = semantic_color_all_frames_map[iter_time_idx]
                # Loss for current frame
                loss, variables, losses = get_loss_gs(params, iter_data, variables, config['train']['loss_weights'],
                                                      load_semantics=load_semantics)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['train']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['train']['densify_dict'],
                                                    params_opt_exclude)
                        if config['use_wandb']:
                            wandb_run.log({"Number of Gaussians - Densification": params['means3D'].shape[0]})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['train']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['train']['sil_thres'], 
                                            mapping=True, load_semantics=load_semantics, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                    # Eval Params at 7K Iterations
                    if (iter + 1) == 7000:
                        print("Evaluating Params at 7K Iterations")
                        eval_params = convert_params_to_store(params)
                        output_dir = os.path.join(config["workdir"], config["run_name"])
                        eval_dir = os.path.join(output_dir, "eval_7k")
                        os.makedirs(eval_dir, exist_ok=True)
                        if config['use_wandb']:
                            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'], wandb_run=wandb_run,
                                 wandb_save_qual=config['wandb']['eval_save_qual'], mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True,
                                 load_semantics=load_semantics)
                        else:
                            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True, load_semantics=load_semantics)
            if num_iters_mapping > 0:
                progress_bar.close()

        # Increment WandB Step
        if config['use_wandb']:
            wandb_time_step += 1

    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate Final Parameters
    with torch.no_grad():
        eval_params = convert_params_to_store(params)
        if config['use_wandb']:
            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'], wandb_run=wandb_run,
                 wandb_save_qual=config['wandb']['eval_save_qual'], mapping_iters=config["train"]["num_iters_mapping"],
                 add_new_gaussians=True, load_semantics=load_semantics)
        else:
            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True, load_semantics=load_semantics)

    # Add Camera Parameters to Save them
    params = eval_params
    params['timestep'] = variables['timestep']
    params['intrinsics'] = map_intrinsics.detach().cpu().numpy()
    params['w2c'] = w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames_map:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)