# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from geort.mocap.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config
import argparse

import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R

# --- setup 3D figure ---
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("MANUS Glove Skeleton")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
ax.view_init(elev=30, azim=30)

axis_len = 0.3
origin = np.array([0, 0, 0])
ax.quiver(*origin, axis_len, 0, 0, color='r', linewidth=2)
ax.quiver(*origin, 0, axis_len, 0, color='g', linewidth=2)
ax.quiver(*origin, 0, 0, axis_len, color='b', linewidth=2)
ax.text(axis_len, 0, 0, 'X', color='r', fontsize=10, weight='bold')
ax.text(0, axis_len, 0, 'Y', color='g', fontsize=10, weight='bold')
ax.text(0, 0, axis_len, 'Z', color='b', fontsize=10, weight='bold')

# pre-create left/right scatter + lines + text labels
left_scatter = ax.scatter([], [], [], color='blue', label='Left Glove')
right_scatter = ax.scatter([], [], [], color='red', label='Right Glove')
left_lines = [ax.plot([], [], [], 'b-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
right_lines = [ax.plot([], [], [], 'r-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
left_texts = [ax.text(0, 0, 0, '', color='blue', fontsize=7) for _ in range(25)]
right_texts = [ax.text(0, 0, 0, '', color='red', fontsize=7) for _ in range(25)]
ax.legend()

# --- helper functions ---
def update_skeleton(floats, scatter, lines, texts, color, ax, frame_len=0.005, offset = np.array([0, 0, 0])):
    pos = floats[:, :3]
    
    # pos[:, 0] = -pos[:, 0]  # mirror X for handedness correction
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

    # update finger lines
    for i in range(5):
        # finger = np.vstack((pos[0], pos[1 + i * 5:1 + (i + 1) * 5]))
        finger = np.vstack((pos[0], pos[i * 5:(i + 1) * 5]))
        lines[i].set_data(finger[:, 0], finger[:, 1])
        lines[i].set_3d_properties(finger[:, 2])

    # update labels
    for i, txt in enumerate(texts):
        txt.set_position((pos[i, 0], pos[i, 1]))
        txt.set_3d_properties(pos[i, 2])
        txt.set_text(str(i))

def parse_full_skeleton(floats):
    update_skeleton(floats, right_scatter, right_lines, right_texts, 'red', ax, offset = np.array([0.1, 0, 0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')
    parser.add_argument('-ckpt_tag', type=str, default='alex')
    parser.add_argument('-data', type=str, default='human')

    args = parser.parse_args()

    # GeoRT Model.
    model = load_model(args.ckpt_tag)
    
    # Motion Capture.
    mocap = ReplayMocap(args.data)
    
    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()
    
    # Run!
    while True:
        for i in range(1):
            viewer_env.update()

        result = mocap.get()

        if result['status'] == 'recording' and result["result"] is not None:
            qpos = model.forward(result["result"])
            hand.set_qpos_target(qpos)

            parse_full_skeleton(result["result"])


        if result['status'] == 'quit':
            break 
        
        plt.pause(0.001)
        time.sleep(0.001)


if __name__ == '__main__':
    main()
