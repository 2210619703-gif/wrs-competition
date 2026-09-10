import pickle


class Grasp(object):

    def __init__(self, ee_values=None, ac_pos=None, ac_rotmat=None):
        self.ee_values = ee_values
        self.ac_pos = ac_pos
        self.ac_rotmat = ac_rotmat

    def __str__(self):
        return "Grasp at " + repr(self.ac_pos) + "\n\t\t" + repr(self.ac_rotmat) + " with " + str(self.ee_values)


class GraspCollection(object):
    def __init__(self, end_effector=None, grasp_list=None):
        self.end_effector = end_effector
        if grasp_list is None:
            self._grasp_list = []
        else:
            self._grasp_list = grasp_list

    @classmethod
    def load_from_disk(cls, file_name="reference_grasp_collection.pickle"):
        with open(file_name, 'rb') as file:
            obj = pickle.load(file)
            if not isinstance(obj, cls):
                raise TypeError(f"Object in {file_name} is not an instance of {cls.__name__}")
            return obj

    def append(self, grasp):
        self._grasp_list.append(grasp)

    def pop(self, index=None):
        if index is None:
            return self._grasp_list.pop()
        return self._grasp_list.pop(index)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._grasp_list[index]
        elif isinstance(index, list):
            return [self._grasp_list[i] for i in index]
        else:
            raise Exception("Index type not supported.")

    def __setitem__(self, index, grasp):
        self._grasp_list[index] = grasp

    def __len__(self):
        return len(self._grasp_list)

    def __iter__(self):
        return iter(self._grasp_list)

    def __add__(self, other):
        if self.end_effector is other.end_effector:
            self._grasp_list += other._grasp_list
            return self
        else:
            raise ValueError("End effectors are not the same.")

    def __str__(self):
        out_str = "GraspCollection of: " + self.end_effector.name if self.end_effector is not None else "" + "\n"
        if len(self._grasp_list) == 0:
            out_str += "  No grasps in the collection.\n"
            return out_str
        for grasp in self._grasp_list:
            out_str += "  " + str(grasp) + "\n"
        return out_str

    def save_to_disk(self, file_name='reference_grasp_collection.pickle'):
        """
        :param reference_grasp_collection:
        :param file_name:
        :return:
        author: haochen, weiwei
        date: 20200104, 20240319
        """
        with open(file_name, 'wb') as file:
            pickle.dump(GraspCollection(grasp_list=self._grasp_list), file)


def grasp_world_to_object_frame(grasp, obj_pos, obj_rotmat):
    """
    将 ``grip_at_by_*`` 产生的、作用于 **世界系** 下的 acting center 位姿
    转为物体系，供 ``wrs.manipulation.pick_place_planner.PickPlacePlanner`` 使用
   （其用 ``obj_rotmat @ ac_pos + obj_pos`` 计算抓取 TCP，故 ``ac_pos``/``ac_rotmat`` 须在物体坐标系中）。

    与 ``EEInterface.hold`` 中 ``rm.rel_pose(ee, object)`` 的分解一致；离线 pickle
    常在物体置於原点、``rotmat=I`` 时生成，物体系与宇宙系数值相同，故与 cobotta 流程兼容。
    """
    o_pos = obj_rotmat.T @ (grasp.ac_pos - obj_pos)
    o_rot = obj_rotmat.T @ grasp.ac_rotmat
    return Grasp(ee_values=grasp.ee_values, ac_pos=o_pos, ac_rotmat=o_rot)


def grasp_collection_world_to_object_frame(grasp_collection, obj_pos, obj_rotmat):
    out = GraspCollection(end_effector=grasp_collection.end_effector)
    for g in grasp_collection:
        out.append(grasp_world_to_object_frame(g, obj_pos, obj_rotmat))
    return out
