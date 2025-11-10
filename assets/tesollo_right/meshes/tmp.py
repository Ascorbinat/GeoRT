import mujoco
import mujoco.viewer

import io
import time

model = mujoco.MjModel.from_xml_path("/home/valentin/git/postdoc/tesollo/DELTO_M_ROS2/dg_description/urdf/for_mujoco/dg5f_right_relative_paths.urdf")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
viewer.sync()

while viewer.is_running():
    viewer.sync()
    time.sleep(0.01)
