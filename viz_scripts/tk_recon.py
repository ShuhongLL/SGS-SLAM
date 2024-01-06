import argparse
import os
import sys
import json
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from queue import Queue
from enum import Enum


from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation
from viz_scripts.final_recon import load_camera, load_scene_data, make_lineset, render, rgbd2pcd

# Queue for inter-thread communication
command_queue = Queue()

class Operation(Enum):
    SWITCH_MODE = 1
    APPLY_MASK = 2


class ControlPanel(tk.Frame):
    def __init__(self, master, command_queue, cfg, width, height, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.master = master
        self.command_queue = command_queue
        master.title('Control Panel')

        # Semantic IDs
        self.semantic_ids_label = tk.Label(master, text='Semantic IDs:')
        self.semantic_ids_label.pack()

        with open(cfg['color_dict_path'], 'r') as file:
            self.color_dict = json.load(file)
        self.sence_name = cfg['scene_name']
        self.semantic_ids = set(self.color_dict[self.sence_name].keys())

        self.semantic_ids_value = tk.Label(master, text=', '.join(str(num) for num in sorted(self.semantic_ids)))
        self.semantic_ids_value.pack()

        # Mode
        self.mode_label = tk.Label(master, text='Mode:')
        self.mode_label.pack()

        self.mode_var = tk.StringVar(value='Colors')  # Default value
        self.colors_radiobutton = tk.Radiobutton(master, text='Colors', variable=self.mode_var, value='color')
        self.colors_radiobutton.pack(anchor=tk.CENTER)

        self.centers_radiobutton = tk.Radiobutton(master, text='Centers', variable=self.mode_var, value='centers')
        self.centers_radiobutton.pack(anchor=tk.CENTER)

        self.semantic_colors_radiobutton = tk.Radiobutton(master, text='Semantic Colors', variable=self.mode_var, value='semantic_color')
        self.semantic_colors_radiobutton.pack(anchor=tk.CENTER)

        # Manipulate
        self.manipulate_label = tk.Label(master, text='Manipulate:')
        self.manipulate_label.pack()

        self.semantic_id_entry_label = tk.Label(master, text='Semantic ID:')
        self.semantic_id_entry_label.pack()

        self.semantic_id_entry = tk.Entry(master)
        self.semantic_id_entry.pack()

        self.keep_var = tk.BooleanVar(value=True)
        self.keep_checkbox = tk.Checkbutton(master, text='Keep', variable=self.keep_var)
        self.keep_checkbox.pack()

        # Apply / Reset buttons
        self.apply_button = tk.Button(master, text='Apply', command=self.apply)
        self.apply_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(master, text='Reset', command=self.reset)
        self.reset_button.pack(side=tk.RIGHT)

    def apply(self):
        # Here you would handle the application of the settings
        mode = self.mode_var.get()
        semantic_id = self.semantic_id_entry.get()
        keep = self.keep_var.get()
        # messagebox.showinfo('Apply', f'Applied settings:\nMode: {mode}\nSemantic ID: {semantic_id}\nKeep: {keep}')
        cmd = {'type': Operation.SWITCH_MODE, 'payload': {'mode': mode}}
        print("Apply: ", cmd)
        self.command_queue.put(cmd)
        if semantic_id.isdigit() and semantic_id in self.semantic_ids:
            cmd = {'type': Operation.APPLY_MASK, 'payload': {
                'semantic_id': int(semantic_id) if semantic_id != '' else -1,
                'to_keep': keep
                }
            }
            print("Apply: ", cmd)
            self.command_queue.put(cmd)
        else:
            print("Invalid Apply: ", semantic_id)

    def reset(self):
        # Here you would handle the resetting of the settings to their defaults
        self.mode_var.set('Colors')
        self.semantic_id_entry.delete(0, tk.END)
        self.keep_var.set(False)
        # messagebox.showinfo('Reset', 'Settings have been reset to default values.')
        self.command_queue.put({
            'type': Operation.SWITCH_MODE,
            'payload': {
                'mode': 'color',
            }
        })
        self.command_queue.put({
            'type': Operation.APPLY_MASK,
            'payload': {
                'semantic_id': -2,
            }
        })

def visualize(scene_path, cfg):
    # Load Scene Data
    w2c, k = load_camera(cfg, scene_path)

    if 'load_semantics' in cfg:
        load_semantics = cfg['load_semantics']
    else:
        load_semantics = False

    scene_data, scene_depth_data, scene_semantic_data, all_w2cs, semantic_ids = load_scene_data(
        scene_path, w2c, k, load_semantics=load_semantics)
    
    # Points to keep
    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()

    # vis.create_window()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
        
        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # if load_semantics:
    #     semantic_exclude = [32] #[38]
    #     semantic_exclude = torch.tensor(semantic_exclude).cuda()
    #     for to_remove_id in semantic_exclude:
    #         render_mask &= (semantic_ids.squeeze() != to_remove_id)

    render_mode = cfg['render_mode']

    # Interactive Rendering
    while True:
        # Check for commands from Tkinter
        while not command_queue.empty():
            msg = command_queue.get()
            if msg['type'] == Operation.SWITCH_MODE:
                render_mode = msg['payload']['mode']
            elif load_semantics and msg['type'] == Operation.APPLY_MASK:
                input_semantic_id = int(msg['payload']['semantic_id'])
                if input_semantic_id == -2: # reset
                    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
                elif input_semantic_id == -1: # no action
                    continue
                else:
                    to_keep = msg['payload']['to_keep']
                    render_mask[semantic_ids.squeeze() == input_semantic_id] = bool(to_keep)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if render_mode == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'][render_mask].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'][render_mask].contiguous().double().cpu().numpy())
        elif render_mode == 'semantic_color':
            seg, depth, sil = render(w2c, k, scene_semantic_data, scene_depth_data, cfg, render_mask)
            pts, cols = rgbd2pcd(seg, depth, w2c, k, cfg)
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    # Start the Open3D visualizer in a separate thread
    thread = Thread(target=visualize, args=(scene_path, viz_cfg))
    thread.daemon = True
    thread.start()

    # Create the Tkinter window
    root = tk.Tk()

    # Create the control panel
    control_panel = ControlPanel(root, command_queue, viz_cfg, width=20, height=200)

    # Run the application
    root.mainloop()
