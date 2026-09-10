"""
Compatibility entry for historical `Lite6WRSGripper`.

This module now maps the legacy class name to the maintained V4
implementation so old import paths keep working.
"""
import numpy as np
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v4 import WRSGripper4


class Lite6WRSGripper(WRSGripper4):
    """Alias of `WRSGripper4` under the historical class name."""

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name='wrs_gripper',
                 enable_cc=True):
        # keep `enable_cc` in the signature for backward compatibility
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    mgm.gen_frame().attach_to(base)
    grpr = Lite6WRSGripper()
    grpr.change_jaw_width(.08)
    grpr.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    base.run()
