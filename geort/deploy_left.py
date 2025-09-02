

import math
import threading
import time
import rclpy
# from allegro_hand_controllers.allegro_robot import AllegroRobot
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time 
from sensor_msgs.msg import JointState
from datetime import datetime
import os 
import zmq
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import geort


import zenoh
from manus import ManusHandData, ManusHandType


def generate_current_timestring():
    """
    Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


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
        # print("Pos", positions.shape, orientation.shape)
        thumb_chain = [0, 1, 2, 3, 4]
        index_chain = [0, 5, 6, 7, 8]
        middle_chain = [0, 9, 10, 11, 12]
        ring_chain = [0, 13, 14, 15, 16]
        pinky_chain = [0, 17, 18, 19, 20]

        # pos 1: posistion of 1 with respect to its parent (0), 
        all_chains = [thumb_chain, index_chain, middle_chain, ring_chain, pinky_chain]

        all_keypoints = {}
        
        # Tw1_w2 * T_w2_w3 * x_w3p
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
    

ALLEGRO_JOINT_NAMES = [f"allegro_joint_{i}.0" for i in range(16)]


class Manus(Node):
    def __init__(self, human_ik, action_scale=0.1, use_8p_retarget=0):
        super().__init__("manus_subscriber")
        # self.allegro_robot = allegro_robot
        self.x_axis = []
        self.y_axis = []
        self.z_axis = []
        self.lock = threading.Lock()

        self.quat = None

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

        self.session = zenoh.open(zenoh.Config())
        self.session.declare_subscriber("manus/hand", self.listener_callback_quat)

        # self.lesft_pub = self.create_publisher(JointState, "/hand/left/joint_cmd", qos)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        # self.right_pub = self.create_publisher(JointState, "/hand/right/joint_cmd", qos)
        self.left_pub = self.create_publisher(JointState, "/hand/left/joint_cmd", qos)
        
        # Our code 
        self.use_8p_retarget = use_8p_retarget
        self.human_ik = human_ik
        self.joint_pos = None

        self.last_action = np.zeros(16)
        self.last_effective_action = np.zeros(16)
        
        self.subscriber =  self.create_subscription(
            JointState,
            "/hand/left/joint_states",
            self.on_allegro_joint_angles,
            1,
        )

        self.hand_init_qpos = {
            "joint_0.0": -0.2182,
            "joint_1.0": 1.4387,
            "joint_2.0": 0.0433,
            "joint_3.0": 0.3360,
            "joint_12.0": 1.3931,  # 0.600,
            "joint_13.0": 0.5550,  # 1.1630,
            "joint_14.0": 0.8481,  # 1.000,
            "joint_15.0": -0.1614,  # 0.480,
            "joint_4.0": -0.1438,
            "joint_5.0": 0.8644,
            "joint_6.0": 0.5840,
            "joint_7.0": -0.0709,
            "joint_8.0": 0.0482,
            "joint_9.0": 1.3183,
            "joint_10.0": 0.5261,
            "joint_11.0": -0.1277,
        }
        self.init_cmd = np.array([self.hand_init_qpos[f"joint_{i}.0"] for i in range(16)])

        self.joint_limit = {
            "joint_0.0":  [-0.47, 0.47],
            "joint_1.0":  [-0.196, 1.61],
            "joint_2.0":  [-0.174, 1.709],
            "joint_3.0":  [-0.227, 1.618],
            "joint_4.0":  [-0.47, 0.47],
            "joint_5.0":  [-0.196, 1.61],
            "joint_6.0":  [-0.174, 1.709],
            "joint_7.0":  [-0.227, 1.618],
            "joint_8.0":  [-0.47, 0.47],
            "joint_9.0":  [-0.196, 1.61],
            "joint_10.0": [-0.174, 1.709],
            "joint_11.0": [-0.227, 1.618],
            "joint_12.0": [0.263, 1.396],
            "joint_13.0": [-0.105, 1.163],
            "joint_14.0": [-0.189, 1.644],
            "joint_15.0": [-0.162, 1.719],
        }

        self.last_joint_t = time.time()
        self.joint_order = [f"joint_{i}.0" for i in range(16)]
        self.joint_lower_bound = np.array([self.joint_limit[n][0] for n in self.joint_order])
        self.joint_upper_bound = np.array([self.joint_limit[n][1] for n in self.joint_order])


    def on_allegro_joint_angles(self, joint_state):
        joint_angles = np.array(joint_state.position).reshape(16)
        self.get_logger().debug(f"Received joint positions: {joint_angles}")
        self.joint_pos = joint_angles 
        self.last_joint_t = time.time()

        return 

    def listener_callback_quat(self, sample):
        msg = ManusHandData().parse(sample.payload.to_bytes())

        with self.lock:
            if msg.hand == ManusHandType.LEFT:
                self.quat = np.array([[q.x, q.y, q.z, q.w] for q in msg.quaternions])
            elif msg.hand == ManusHandType.RIGHT:
                # self.quat = np.array([[q.x, q.y, q.z, q.w] for q in msg.quaternions])
                pass
      

    def run(self):
        count = 0
    
        current_qpos = None
        last_delta = np.zeros(16)
        # we will need to init everything.
        while current_qpos is None:
            if self.joint_pos is not None:
                current_qpos = np.array(self.joint_pos) 
                current_target = current_qpos
            else:
                time.sleep(1)
            print("Waiting init qpos to start")

        kinematics_solver = ManusForwardKinematicsSolver()

        while rclpy.ok():
            last_time = time.time()
            count = count + 1
            # print("Running...", len(self.quat))
            
            if self.quat is None:
                continue
            # if self.pos is None or self.quat is None or len(self.x_axis) == 0:
            #     continue
            
            keypoints = kinematics_solver.solve_keypoints(self.pos, self.quat)
            keypoints =  np.array([keypoints[i] for i in range(21)])
            keypoints = hand_to_canonical(keypoints)
            # keypoints[1:5,0] *= -1

            joint_cmd = self.human_ik.forward(keypoints) # TODO(): use GeoRT. check API consistency.
            
            delta = joint_cmd - current_target
            delta = np.clip(delta, -0.1, 0.1)
        
            current_target = current_target + delta
            current_target = np.clip(current_target, self.joint_lower_bound, self.joint_upper_bound)

            # self.allegro_robot.command_joint_position(current_target)

            # command to allegro roboto only right atm

            now = self.get_clock().now().to_msg()
            hand_left_msg = JointState()
            hand_left_msg.header.stamp = now
            hand_left_msg.name = ALLEGRO_JOINT_NAMES
            hand_left_msg.position = current_target.tolist()
            self.left_pub.publish(hand_left_msg)

            # print("Current target: ", current_target)
            current_time = time.time()
            yield_time = (last_time + 0.01) - current_time
            if yield_time > 0:
                time.sleep(yield_time)
            # print("T", time.time() - last_time)


    def real_to_isaac(self, x):
        # In Isaac: [0, 1, 2, 3], [12, 13, 14, 15], [4, 5, 6, 7], [8, 9, 10, 11]
        if len(x.shape) == 1:
            return np.concatenate([x[0:4], x[12:16], x[4:8], x[8:12]], axis=0)
        else:
            return np.concatenate([x[:, 0:4], x[:, 12:16], x[:, 4:8], x[:, 8:12]], axis=-1)


    def isaac_to_real(self, x):
        if len(x.shape) == 1:
            return np.concatenate([x[0:4], x[8:12], x[12:16], x[4:8]], axis=0)
        else:
            return np.concatenate([x[:, 0:4], x[:, 8:12], x[:, 12:16], x[:, 4:8]], axis=0)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='krish_left', type=str)
    parser.add_argument('-e', type=int)
    pargs = parser.parse_args()

    rclpy.init(args=args)

    print(f"Loading model: {pargs.model}, epoch: {pargs.e}")

    human_ik = geort.load_model(pargs.model, epoch=pargs.e) 
    # allegro_node = AllegroRobot(hand_topic_prefix="allegroHand")
    
    manus_node = Manus(human_ik)

    executor = rclpy.executors.SingleThreadedExecutor()
    # executor.add_node(allegro_node)
    executor.add_node(manus_node)

    # Create a thread for the run method
    run_thread = threading.Thread(target=manus_node.run)
    run_thread.start()

    executor.spin()
    run_thread.join()  # Wait for the run thread to finish

if __name__ == "__main__":
    main()
