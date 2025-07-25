#!/usr/bin/env python3
# filename: play_smpl_rots.py

import os
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from smplx import SMPL
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_rotations(np_file: str):
    rots = np.load(np_file)                # (T, 24, 3, 3)
    T, J = rots.shape[:2]
    R = rots.reshape(-1, 3, 3)             # (T*24, 3, 3)
    r = Rotation.from_matrix(R)
    rotvecs = r.as_rotvec().reshape(T, J, 3)
    poses   = rotvecs.reshape(T, J*3)      # (T, 72)
    return poses

def compute_joints(poses: np.ndarray, model_folder: str, gender: str):
    device = torch.device('cpu')
    model = SMPL(model_folder, gender=gender, batch_size=1).to(device)
    T = poses.shape[0]
    joints_all = []

    for t in range(T):
        go = torch.from_numpy(poses[t,:3]).unsqueeze(0).float().to(device)
        bp = torch.from_numpy(poses[t,3:]).unsqueeze(0).float().to(device)
        out = model(global_orient=go,
                    body_pose=bp,
                    betas=torch.zeros(1, model.betas.shape[-1], device=device),
                    transl=torch.zeros(1,3,device=device))
        joints_all.append(out.joints.detach().cpu().numpy()[0])  # (24,3)

    return np.stack(joints_all)  # (T, 24, 3)

def make_animation(joints_all: np.ndarray,
                   edges: list[tuple[int,int]],
                   save_gif: bool = True,
                   gif_path: str = "skeleton.gif"):
    T = joints_all.shape[0]
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

    def update(frame):
        ax.cla()
        pts = joints_all[frame][:, [0, 2, 1]].copy()
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=20, c='k')
        for a, b in edges:
            ax.plot(
                [pts[a,0], pts[b,0]],
                [pts[a,1], pts[b,1]],
                [pts[a,2], pts[b,2]],
                lw=2, c='r'
            )
        ax.set_title(f"Frame {frame+1}/{T}")
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        return []

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)

    if save_gif:
        writer = PillowWriter(fps=20)
        ani.save(gif_path, writer=writer)
        print(f"Saved GIF to {gif_path}")

    plt.show()

def main():
    np_file      = "cache/side_flip_pose.npy"
    model_folder = "deps/smpl_models/smpl"
    gender       = "MALE"

    if not os.path.exists(np_file):
        raise FileNotFoundError(f"{np_file} not found.")
    if not os.path.isdir(model_folder):
        raise NotADirectoryError(f"{model_folder} is not a directory.")

    poses      = load_rotations(np_file)
    joints_all = compute_joints(poses, model_folder, gender)

    # SMPL joint 연결 구조 (예: pelvis→hip→knee→ankle 등)
    edges = [
        (0,1),(1,4),(4,7),(7,10),      # 왼쪽 다리
        (0,2),(2,5),(5,8),(8,11),      # 오른쪽 다리
        (0,3),(3,6),(6,9),(9,12),(12,15), # 척추→목→머리
        (9,13),(13,16),(16,18),(18,20),(20,22), # 왼쪽 팔
        (9,14),(14,17),(17,19),(19,21),(21,23)  # 오른쪽 팔
    ]

    make_animation(joints_all, edges, save_gif=True, gif_path="side_flip.gif")

if __name__ == "__main__":
    main()
