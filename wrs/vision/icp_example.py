import numpy as np

from wrs import wd, rm, mgm, mcm, cbt
import wrs.basis.robot_math as rm
from wrs.vision.depth_camera.util_functions import registration_ptpt
base = wd.World(cam_pos=rm.vec(1.2, .7, 1), lookat_pos=rm.vec(.0, 0, .15))
mgm.gen_frame().attach_to(base)
# ground
shoe = mgm.GeometricModel(r'D:/Project/wrs-main-htw/0000_examples/objects/tiaozhanbei/shoes.stl')
shoe.attach_to(base)
shoe_pcd = shoe.sample_surface(radius=0.001,n_samples=10000)
shoe_pcd = shoe_pcd[shoe_pcd[:,2] > .018]
# 移除这一行，因为我们希望分开展示原始点云和真实点云
# mgm.gen_pointcloud(shoe_pcd).attach_to(base)


import pickle
with open('cup_scene_world_color.pkl', 'rb') as f:
    pcd_data = pickle.load(f)

shoe_pcd_real = pcd_data['points']

# 将原始点云 (shoe_pcd) 设计成绿色点云
# mgm.gen_pointcloud(shoe_pcd,rgba=np.array([0,1,0,1])).attach_to(base) # 绿色 (R=0, G=1, B=0, A=1)

shoe_pcd_real = shoe_pcd_real[(shoe_pcd_real[:,2]>0.02) & (shoe_pcd_real[:,2]<0.09) & (shoe_pcd_real[:,0]<0.5)
& (shoe_pcd_real[:,1]<-0.05) & (shoe_pcd_real[:,1]>-0.4) ]

# 将真实点云 (shoe_pcd_real) 设计成黄色点云，以便与匹配结果区分
mgm.gen_pointcloud(shoe_pcd_real,rgba=np.array([1,1,0,0.5])).attach_to(base) # 黄色 (R=1, G=1, B=0, A=0.5)

# rotmat = rm.rotmat_from_axangle(ax=rm.unit_vector(np.array([1,0.3,0.7])), angle=np.radians(66))
# position = np.array([.3 , .2, .1,])
# shoe_pcd2 = (rotmat.dot(shoe_pcd.T)).T
# shoe_pcd2 = shoe_pcd2 + position
# mgm.gen_pointcloud(shoe_pcd2, rgba=np.array([1,0,0,1])).attach_to(base)
# print("rotmat", rotmat)
# print('position', position)
icp_result = registration_ptpt(shoe_pcd, shoe_pcd_real, downsampling_voxelsize=.07, )
print(icp_result)
transformation = icp_result[2]
shoe.homomat = transformation
# shoe.attach_to(base)
show_pcd2= rm.transform_points_by_homomat(transformation, shoe_pcd.copy())

# 将匹配后的点云 (show_pcd2) 设计成红色点云
mgm.gen_pointcloud(show_pcd2, rgba=np.array([1,0,0,1])).attach_to(base) # 红色 (R=1, G=0, B=0, A=1)

base.run()