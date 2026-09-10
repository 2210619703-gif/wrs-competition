"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate the camera
"""
__VERSION__ = '0.0.1'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod

from direct.task.TaskManagerGlobal import taskMgr

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri
import numpy as np


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


class ManualCalibrationBase(ABC):
    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 component_name="arm", move_resolution=.005, rotation_resolution=rm.np.radians(5)):
        """
        Class to manually calibrate the point cloud data
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix. If it is None, the init calibration matrix will be identity matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = np.eye(4) if init_calib_mat is None else init_calib_mat
        self._component_name = component_name

        # variable stores robot plot and the point cloud plot
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        #
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # add task
        taskMgr.doMethodLater(5, self.sync_rbt, "sync rbt", )
        taskMgr.add(self.adjust, "manual adjust the mph")
        # taskMgr.doMethodLater(5 ,self.sync_pcd, "sync mph", )
        # taskMgr.add(self.sync_rbt_once, "sync_rbt_once")
        # taskMgr.add(self.adjust, "manual adjust the mph")
        # taskMgr.add(self.sync_pcd_once, "sync_pcd_once")

    @abstractmethod
    def get_pcd(self) -> np.ndarray:
        """
        An abstract method to get the point cloud
        :return: An Nx3 ndarray represents the point cloud
        """
        pass

    @abstractmethod
    def get_rbt_jnt_val(self) -> np.ndarray:
        """
        An abstract method to get the robot joint angles
        :return: 1xn ndarray, n is degree of freedom of the robot
        """
        pass

    @abstractmethod
    def align_pcd(self, pcd) -> np.ndarray:
        """
        Abstract method to align the mph according to the calibration matrix
        implement the Eye-in-hand or eye-to-hand transformation here
        https://support.zivid.com/en/latest/academy/applications/hand-eye/system-configurations.html
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    # def manual_refresh_pcd(self,task):
    def manual_refresh_pcd(self):
        """手动触发点云刷新"""
        print("Refreshing point cloud")
        self.sync_rbt(None)
        self.sync_pcd(None)  # 直接调用sync_pcd
        # return task.again

    def sync_pcd_once(self, task):
        """执行后立即移除任务"""
        self._pcd = self.get_pcd()
        self.plot()
        self.save()
        # return task.done

    def sync_rbt_once(self, task):
        """执行后立即移除任务"""
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(rbt_jnt_val, update=True)
        self.plot()
        # return task.done

    def move_adjust(self, dir, dir_global, key_name=None):
        # 直接在全局世界坐标系下对相机进行平移微调
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        # 直接在全局世界坐标系下计算旋转增量矩阵
        delta_R_world = rm.rotmat_from_axangle(dir_global, np.radians(self.rotation_resolution))
        current_R_w_c = self._init_calib_mat[:3, :3]
        # 左乘 delta_R_world 意味着围绕世界坐标系(或基坐标系)的坐标轴进行旋转
        self._init_calib_mat[:3, :3] = np.dot(delta_R_world, current_R_w_c)

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n', refresh='r'):
        def add_key(keys: str or list):
            """
            Add key to  the keymap. The default keymap can be seen in visualization/panda/inputmanager.py
            :param keys: the keys added to the keymap
            """
            assert isinstance(keys, str) or isinstance(keys, list)

            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw

        self._key['refresh'] = 'r'  # 例如按 'r' 键刷新
        base.inputmgr.accept('r', self.manual_refresh_pcd)

    def sync_pcd(self, task):
        """
        Synchronize the real robot and the simulation robot
        :return: None
        """

        self._pcd = self.get_pcd()
        self.plot()
        self.save()
        # return task.again

    def sync_rbt(self, task):
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(rbt_jnt_val, update=True)
        self.plot()
        # return task.again

    def save(self):
        """
        Save manual calibration results
        :return:
        """
        dump_json({'affine_mat': self._init_calib_mat.tolist()},
                  r"manual_calibration_piper_outhand.json",
                  reminder=False)

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot
        :param task:
        :return:
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=0.3)
        self._plot_node_rbt.attach_to(base)
        pcd = self._pcd
        if pcd is not None:
            if pcd.shape[1] == 6:
                pcd, pcd_color = pcd[:, :3], pcd[:, 3:6]
                pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = np.array([1, 1, 1, 1])
            pcd_r = self.align_pcd(pcd)
            self._plot_node_pcd = mgm.gen_pointcloud(pcd_r, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
        # if task is not None:
        # return task.again

    def adjust(self, task):
        """
        Checks for keyboard input and adjusts the calibration matrix.
        This method is optimized to prevent lag by only plotting when a change occurs
        and by removing expensive I/O operations from the loop.
        """
        was_adjusted = False

        # --- Handle Translation ---
        # Use separate 'if' statements to allow for simultaneous key presses (e.g., diagonal movement)
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_')
            was_adjusted = True

        # --- Handle Rotation ---
        if base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_ccw')
            was_adjusted = True

        # --- Update Scene only if something changed ---
        if was_adjusted:
            self.plot()
            self.save()
            # IMPORTANT: We have removed self.save() from this loop.
            # Call self.save() manually when you are done with the calibration.

        return task.again


class RealmanManualCalib(ManualCalibrationBase):
    """
    Eye in hand example
    """

    def get_pcd(self):
        pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth()

        return np.hstack((pcd, pcd_color))

    def get_rbt_jnt_val(self):
        return self._rbt_x.get_joint_values()

    def align_pcd(self, pcd):
        # 眼在手外：self._init_calib_mat 直接就是相机到世界(机械臂基座)的矩阵 w2c_mat
        w2c_mat = self._init_calib_mat
        return rm.transform_points_by_homomat(w2c_mat, points=pcd)


def load_calibration_matrix_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    mat_list = data.get('affine_mat', None)
    if mat_list is None:
        raise ValueError("JSON 文件中没有找到 'affine_mat' 字段")
    return np.array(mat_list)


# middle:
#   ID: '243322073422'
# left:
#   ID: '243322074546'
# right:
#   ID: '243322071033'
if __name__ == "__main__":
    import numpy as np
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.robot_con.piper.piper import PiperArmController
    from wrs.robot_sim.manipulators.piper.piper import Piper
    from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
    from wrs import mcm

    base = wd.World(cam_pos=[0, 2, 0], lookat_pos=[0, 0, 0], lens_type=2)

    # 左臂base与世界坐标系重合
    rbt_left = PiperSglArm(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0), name='piper_left')

    # 加载左臂
    rbtx_left = PiperArmController(can_name="can0", has_gripper=True)

    # 读取三个标定矩阵
    init_mat = load_calibration_matrix_from_json(
        r"manual_calibration_piper_outhand.json")

    mgm.gen_frame(ax_length=1).attach_to(base)

    holeBoard = mcm.CollisionModel(r"holeboard.stl")
    holeBoard.rgba = rm.np.array([0, 0, 255, 0.7])
    h1_gl_pos = rm.np.array([0.245, 0, 0])
    h1_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
    holeBoard.pos = h1_gl_pos
    holeBoard.rotmat = h1_gl_rotmat
    holeBoard.attach_to(base)
    # 左臂相机id
    # cam_ids = '243322074546'

    rs_pipe = RealSenseD400()

    rs_pipe.get_pcd_texture_depth()  # warm-up
    rs_pipe.get_pcd_texture_depth()

    xarm_mc = RealmanManualCalib(rbt_s=rbt_left, rbt_x=rbtx_left, sensor_hdl=rs_pipe,
                                 init_calib_mat=init_mat,
                                 # move_resolution=0.0025,
                                 move_resolution=0.0005,
                                 # rotation_resolution=np.radians(10))
                                 rotation_resolution=np.radians(10))

    # # 加载右臂
    # rbtx_right = PiperArmController(can_name="can1", has_gripper=True)
    #
    # #右臂在world Y轴 -0.6m处
    # rbt_right = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0),pos=[0, -0.6, 0], name='piper_right')
    #
    # # 同步右臂的任务
    # right_arm = None
    #
    # def sync_right_arm(task):
    #     global right_arm
    #     if right_arm is not None:
    #         right_arm.detach()
    #     rbt_jnt_val = rbtx_right.get_joint_values()
    #     rbt_right.fk(rbt_jnt_val, update=True)
    #     right_arm = rbt_right.gen_meshmodel(alpha=1).attach_to(base)
    #     return task.again
    #
    # taskMgr.doMethodLater(0.01, sync_right_arm, "sync right arm")

    base.run()




