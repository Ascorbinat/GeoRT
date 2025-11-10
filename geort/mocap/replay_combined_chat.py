# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from geort.mocap.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config

import argparse
import numpy as np
import time
import multiprocessing as mp
import matplotlib
matplotlib.use("TkAgg")  # safe interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# --- matplotlib setup ---
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

left_scatter = ax.scatter([], [], [], color='blue', label='Left Glove')
right_scatter = ax.scatter([], [], [], color='red', label='Right Glove')
left_lines = [ax.plot([], [], [], 'b-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
right_lines = [ax.plot([], [], [], 'r-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
left_texts = [ax.text(0, 0, 0, '', color='blue', fontsize=7) for _ in range(25)]
right_texts = [ax.text(0, 0, 0, '', color='red', fontsize=7) for _ in range(25)]
ax.legend()

def update_skeleton(floats, scatter, lines, texts, color, ax, frame_len=0.005, offset=np.array([0,0,0])):
    pos = floats[:, :3]
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    for i in range(5):
        finger = np.vstack((pos[0], pos[i * 5:(i + 1) * 5]))
        lines[i].set_data(finger[:, 0], finger[:, 1])
        lines[i].set_3d_properties(finger[:, 2])
    for i, txt in enumerate(texts):
        txt.set_position((pos[i, 0], pos[i, 1]))
        txt.set_3d_properties(pos[i, 2])
        txt.set_text(str(i))

def parse_full_skeleton(floats):
    update_skeleton(floats, right_scatter, right_lines, right_texts, 'red', ax, offset=np.array([0.1,0,0]))


# --- separate process for SAPIEN viewer ---
def sapien_process(q, hand_config, ckpt_tag, data_name):
    """Runs the SAPIEN simulation and listens for qpos updates from the main process."""
    from geort import load_model, get_config
    from geort.env.hand import HandKinematicModel
    import numpy as np
    import time

    model = load_model(ckpt_tag)
    mocap = ReplayMocap(data_name)

    config = get_config(hand_config)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()

    print("[SAPIEN process] Viewer started.")
    while True:
        viewer_env.update()

        # Receive new qpos targets if available
        try:
            while not q.empty():
                msg = q.get_nowait()
                if msg == "quit":
                    print("[SAPIEN process] Received quit.")
                    return
                else:
                    qpos = np.array(msg, dtype=np.float32)
                    hand.set_qpos_target(qpos)
        except Exception:
            pass

        time.sleep(0.005)


# --- main (matplotlib + mocap loop) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')
    parser.add_argument('-ckpt_tag', type=str, default='alex')
    parser.add_argument('-data', type=str, default='human')
    args = parser.parse_args()

    model = load_model(args.ckpt_tag)
    mocap = ReplayMocap(args.data)

    q = mp.Queue()
    sapien_proc = mp.Process(target=sapien_process, args=(q, args.hand, args.ckpt_tag, args.data))
    sapien_proc.start()

    try:
        while True:
            result = mocap.get()
            if result['status'] == 'recording' and result["result"] is not None:
                floats = result["result"]
                qpos = model.forward(floats)
                q.put(qpos.tolist())     # send to sapien process
                parse_full_skeleton(floats)
            elif result['status'] == 'quit':
                break

            plt.pause(0.001)
            time.sleep(0.001)
    finally:
        q.put("quit")
        sapien_proc.join(timeout=2.0)
        plt.close(fig)
        print("[Main] Clean shutdown.")

if __name__ == '__main__':
    mp.set_start_method("spawn")  # ensure clean start on Linux/macOS
    main()

