import os
import argparse
import numpy as np
import random
import time
from termcolor import colored
import h5py
import zarr
from termcolor import cprint
from tqdm import tqdm
import json
import open3d as o3d  # Import open3d for PLY handling
import matplotlib.pyplot as plt


def convert_dataset(args):
    demo_dir = args.demo_dir
    save_dir = args.save_dir
    
    save_img = args.save_img
    save_depth = args.save_depth
    
    # create dir to save demonstrations
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    
    episode_dirs = [f for f in os.listdir(demo_dir) if f.startswith("episode")]
    episode_dirs = sorted(episode_dirs)
    
    total_count = 0
    color_arrays = []
    depth_arrays = []
    cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    
    for episode_dir in episode_dirs:
        # load episode's data.json file
        episode_path = os.path.join(demo_dir, episode_dir, 'data.json')
        with open(episode_path, 'r') as f:
            episode_data = json.load(f)
        
        color_array = []
        depth_array = []
        cloud_array = []
        state_array = []
        action_array = []
        
        # Process every 3rd frame to downsample to 10Hz (assuming 30Hz original frequency)
        for i in tqdm(range(0, len(episode_data), 3), desc=f"Processing {episode_dir}"):
            data = episode_data[i]
            
            # Read state from json
            pos_xyz = np.array(data["pos_xyz"])
            quat_xyzw = np.array(data["quat_xyzw"])
            gripper = np.array([data["gripper"]], dtype=np.float32)
            state = np.concatenate([pos_xyz, quat_xyzw, gripper])

            # Read action from json
            cmd_pos_xyz = np.array(data["cmd_pos_xyz"])
            cmd_quat_xyzw = np.array(data["cmd_quat_xyzw"])
            next_gripper = np.array([episode_data[i+1]["gripper"] if i+1 < len(episode_data) else gripper], dtype=np.float32)
            action = np.concatenate([cmd_pos_xyz, cmd_quat_xyzw, next_gripper])
            
            # Read image, depth and point cloud
            if save_img:
                img_path = os.path.join(demo_dir, episode_dir, 'resized_images', f"{data['index']}.jpg")
                color_image = np.array(plt.imread(img_path))  # Load image
                color_array.append(color_image)
            
            if save_depth:
                depth_img_path = os.path.join(demo_dir, episode_dir, 'depth_images', f"{data['index']}.jpg")
                depth_image = np.array(plt.imread(depth_img_path))  # Load depth image
                depth_array.append(depth_image)

            # Load point cloud from PLY file using open3d
            cloud_path = os.path.join(demo_dir, episode_dir, 'resized_point_clouds', f"{data['index']}.ply")
            pcd = o3d.io.read_point_cloud(cloud_path)  # Read the PLY file
            cloud = np.asarray(pcd.points)  # Convert to numpy array
            
            # Perform point cloud downsampling to 10000 points
            if cloud.shape[0] > 10000:
                selected_idx = np.random.choice(cloud.shape[0], 10000, replace=True)
                cloud = cloud[selected_idx]

            # Append data for current frame
            cloud_array.append(cloud)
            state_array.append(state)
            action_array.append(action)
        
        total_count += len(action_array)
        cloud_arrays.extend(cloud_array)
        
        if save_img:
            color_arrays.extend(color_array)
        
        if save_depth:
            depth_arrays.extend(depth_array)
        
        state_arrays.extend(state_array)
        action_arrays.extend(action_array)
        episode_ends_arrays.append(total_count)

    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    if save_img:
        color_arrays = np.stack(color_arrays, axis=0)
        if color_arrays.shape[1] == 3:  # make channel last
            color_arrays = np.transpose(color_arrays, (0, 2, 3, 1))
    
    if save_depth:
        depth_arrays = np.stack(depth_arrays, axis=0)

    state_arrays = np.stack(state_arrays, axis=0)
    cloud_arrays = np.stack(cloud_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    single_size = 500
    state_chunk_size = (single_size, state_arrays.shape[1])
    point_cloud_chunk_size = (single_size, cloud_arrays.shape[1], cloud_arrays.shape[2])
    action_chunk_size = (single_size, action_arrays.shape[1])
    if save_img:
        img_chunk_size = (single_size, color_arrays.shape[1], color_arrays.shape[2], color_arrays.shape[3])
        zarr_data.create_dataset('img', data=color_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

    if save_depth:
        depth_chunk_size = (single_size, depth_arrays.shape[1], depth_arrays.shape[2])
        zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    # print shape
    if save_img:
        cprint(f'color shape: {color_arrays.shape}, range: [{np.min(color_arrays)}, {np.max(color_arrays)}]', 'green')
    if save_depth:
        cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'cloud shape: {cloud_arrays.shape}, range: [{np.min(cloud_arrays)}, {np.max(cloud_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # count file size
    total_size = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    cprint(f"Total size: {total_size/1e6} MB", "green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_img", type=int)
    parser.add_argument("--save_depth", type=int)
    
    args = parser.parse_args()
    
    convert_dataset(args)
