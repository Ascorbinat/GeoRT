import math
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os 
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import geort 


import zenoh
from manus import ManusHandData, ManusHandType


def nopop_pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

class MatplotlibRenderer:
    def __init__(self):
        # render object holder.
        self.objects = {}

        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.axes.set_xlim3d(left=-0.2, right=0.2) 
        self.ax.axes.set_ylim3d(bottom=-0.2, top=0.2) 
        self.ax.axes.set_zlim3d(bottom=-0.1, top=0.3) 

        return
    
    def add_scatter(self, name, num_points, color='b', marker='.', s=10):
        self.objects[name] = self.ax.scatter(np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), c=color, marker=marker, s=s)
        return
    
    def add_lines(self, name, num_lines):
        self.objects[name] = [self.ax.plot([], [], [])[0] for i in range(num_lines)]
        return 
    
    def update_scatter(self, name, point_data):
        self.objects[name]._offsets3d = (point_data[:, 0], point_data[:, 1], point_data[:, 2])
        return
    
    def update_lines(self, name, line_data):        
        # line_data [[start1, end1], [start2, end2], ...]
        for i, line in enumerate(line_data):
            start, end = line
            self.objects[name][i].set_data([start[0], end[0]], [start[1], end[1]])
            self.objects[name][i].set_3d_properties([start[2], end[2]])
        return
    
    def update(self, t=0.002):
        plt.draw()
        nopop_pause(t)
        return
    
    def clear(self):
        for k, v in self.objects.items():
            v.remove()

        # plt.cla()
        # del self.objects
        plt.draw()
        self.objects = {}


def hand_to_canonical(hand_point):
    z_axis = hand_point[9] - hand_point[0]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = -(hand_point[5] - hand_point[13])
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)


    rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()
   
    tranlation_base = hand_point[0]

    transform = np.eye(4)
    transform[:3, :3] = rotation_base
    transform[:3, 3] = tranlation_base
    
    transform_inv = np.linalg.inv(transform)
    hand_point = np.array(hand_point)
    hand_point = np.concatenate((np.array(hand_point), np.ones((21, 1))), axis=-1)

    hand_point = hand_point @ transform_inv.transpose()
        
    return hand_point[:, :3]


class ManusForwardKinematicsSolver:
    def __init__(self):
        return 


    def make_transformation_matrix(self, pos, quat):
        # create a 4*4 transformation matrix.
        from scipy.spatial.transform import Rotation as R
        out = np.eye(4)
        out[:3, 3] = pos
        out[:3, :3] = R.from_quat(quat).as_matrix()#.transpose()
        return out

    def solve_keypoints(self, positions, orientation):
        # position: the end point with respect to the parent link.
        print("Pos", positions.shape, orientation.shape)
        thumb_chain = [0, 1, 2, 3, 4]
        index_chain = [0, 5, 6, 7, 8]
        middle_chain = [0, 9, 10, 11, 12]
        ring_chain = [0, 13, 14, 15, 16]
        pinky_chain = [0, 17, 18, 19, 20]

        # pos 1: posistion of 1 with respect to its parent (0), 
        all_chains = [thumb_chain, index_chain, middle_chain, ring_chain, pinky_chain]

        all_keypoints = {}
        
        for chain in all_chains:
            current_transformation_to_world = np.eye(4)

            for idx in chain:
                pos = np.array(positions[idx])
                # print(idx,)
                # pos[0] = - pos[0]
                transformation = self.make_transformation_matrix(pos, orientation[idx])

                last_position = np.array(current_transformation_to_world[:3, 3])
                current_transformation_to_world = current_transformation_to_world @ transformation
                position = current_transformation_to_world[:3, 3]
                # print("D", np.linalg.norm(last_position - position) - np.linalg.norm(pos))

                if idx not in all_keypoints:
                    all_keypoints[idx] = position

        return all_keypoints

    def solve_keypoints_raw(self, positions, orientation):
        # position: the end point with respect to the parent link.
        print("Pos", positions.shape, orientation.shape)
        thumb_chain = [0, 1, 2, 3, 4]
        index_chain = [0, 5, 6, 7, 8]
        middle_chain = [0, 9, 10, 11, 12]
        ring_chain = [0, 13, 14, 15, 16]
        pinky_chain = [0, 17, 18, 19, 20]

        # pos 1: posistion of 1 with respect to its parent (0), 
        all_chains = [thumb_chain, index_chain, middle_chain, ring_chain, pinky_chain]

        all_keypoints = {}
        
        for chain in all_chains:
            current_transformation_to_world = np.eye(4)

            for idx in chain:
                pos = np.array(positions[idx])
                # print(idx,)
                # pos[0] = - pos[0]
                transformation = self.make_transformation_matrix(pos, orientation[idx])

                last_position = np.array(current_transformation_to_world[:3, 3])
                current_transformation_to_world = current_transformation_to_world @ transformation
                position = current_transformation_to_world[:3, 3]
                # print("D", np.linalg.norm(last_position - position) - np.linalg.norm(pos))

                if idx not in all_keypoints:
                    all_keypoints[idx] = position

        return all_keypoints


class Manus(Node):
    def __init__(self, data_name="alex"):
        super().__init__("manus_visualizer")
        self.data_name = data_name 
        self.shutdown_event = threading.Event()

        self.x_axis = []
        self.y_axis = []
        self.z_axis = []
        self.pos = None
        self.quat = None
        self.lock = threading.Lock()

        # tuned manus hand min,max joints
        self.hand_joint_angles = [
            (-15, 1),
            (1, 100),
            (1, 110),
            (1, 70),  # Index Finger
            (10, 1),
            (1, 100),
            (1, 110),
            (1, 70),  # Middle Finger
            (10, 1),
            (1, 100),
            (1, 110),
            (1, 70),  # Ring Finger
            (20, -15),
            (40, 0),
            (0, -55),
            (30, -90),  # Thumb Finger
        ]

        # tuned allegro angles
        self.allegro_joint_angles = [
            (0, 0),
            (-11, 98),
            (-10, 98),
            (-13, 98),  # Index Finger
            (0, -0),
            (-11, 98),
            (-10, 98),
            (-13, 98),  # Middle Finger
            (0, -0),
            (-11, 98),
            (-10, 98),
            (-13, 98),  # Ring Finger
            (20, 80),
            (-6, 67),
            (-11, 94),
            (-9, 98),  # Thumb Finger
        ]
        

        self.pos = np.array([
            [0.0, 0.0, 0.0],
            [0.0250, 0.0000, 0.0050],
            [0.0000, 0.0000, 0.0390],
            [0.0000, 0.0000, 0.0330],
            [0.0000, 0.0000, 0.0210],
            [0.0170, 0.0000, 0.0870],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0220],
            [0.0000, 0.0000, 0.0200],
            [0.0000, 0.0000, 0.0920],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0260],
            [0.0000, 0.0000, 0.0220],
            [-0.0170, 0.0000, 0.0840],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0200],
            [-0.0340, 0.0000, 0.0720],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0210],
            [0.0000, 0.0000, 0.0200],
        ])
        print(self.pos.shape)
        # Default Allegro hand joint angles (min, max)


        self.session = zenoh.open(zenoh.Config())
        self.session.declare_subscriber("manus/hand", self.listener_callback_quat)


    def map_joint_angle(self, joint_angle, joint_index):
        # Scale the joint angle from the Manus hand range to the Allegro hand range
        manus_min, manus_max = self.hand_joint_angles[joint_index]
        allegro_min, allegro_max = self.allegro_joint_angles[joint_index]

        # Map the joint angle
        mapped_angle = (joint_angle - manus_min) / (manus_max - manus_min) * (
            allegro_max - allegro_min
        ) + allegro_min

        return mapped_angle


    # def listener_callback_quat(self, msg):
    #     #self.pos_msg = list(msg.data)
    #     #print("POS", self.pos_msg)
    #     self.quat = np.array(list(msg.data)).reshape(21, 4)


    def listener_callback_quat(self, sample):
        msg = ManusHandData().parse(sample.payload.to_bytes())

        with self.lock:
            if msg.hand == ManusHandType.LEFT:
                # self.quat = np.array([[q.x, q.y, q.z, q.w] for q in msg.quaternions])
                pass
                
            elif msg.hand == ManusHandType.RIGHT:
                self.quat = np.array([[q.x, q.y, q.z, q.w] for q in msg.quaternions])
                pass

    def run(self):
        count = 0
        kinematics_solver = ManusForwardKinematicsSolver()

        renderer = MatplotlibRenderer()
        renderer.add_scatter("handv", 21) 

        all_keypoints = []

        count = 1

        while rclpy.ok() and not self.shutdown_event.is_set() and count < 10000:
            count = count + 1
            if self.pos is None or self.quat is None:# or len(self.x_axis) == 0:
                continue

            keypoints = kinematics_solver.solve_keypoints(self.pos, self.quat)
            keypoints =  np.array([keypoints[i] for i in range(21)])
            keypoints = hand_to_canonical(keypoints)
            renderer.update_scatter("handv", keypoints)

            all_keypoints.append(keypoints)
            print("# Data Collected: ", len(all_keypoints))
            renderer.update()


        self.shutdown_event.set()

        data = np.array(all_keypoints)
        geort.save_human_data(data, self.data_name)
        print("GeoRT: Data Saved!", data.shape)


    def stop_collection(self):
        self.get_logger().info("Shutdown signal received, stopping...")
        self.shutdown_event.set()

def main(args=None):
    rclpy.init(args=args)
    data_name = "krish_right" # TODO()
    manus_node = Manus(data_name=data_name)

    # Signal handler
    def signal_handler(sig, frame):
        manus_node.get_logger().info("Ctrl-C pressed! Initiating shutdown...")
        manus_node.stop_collection()


    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(manus_node)

    # Create a thread for the run method
    run_thread = threading.Thread(target=manus_node.run)
    run_thread.start()
    
    try:
        executor.spin()
    finally:
        run_thread.join()

        


if __name__ == "__main__":
    main()
