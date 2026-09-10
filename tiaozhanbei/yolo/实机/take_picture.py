import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
import json


# =====================
# 保存路径
# =====================
save_dir = ("Drywall bugle screws_001")

rgb_dir = os.path.join(save_dir, "rgb")
depth_dir = os.path.join(save_dir, "depth")
pc_dir = os.path.join(save_dir, "pointcloud")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(pc_dir, exist_ok=True)


# =====================
# RealSense初始化
# =====================

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(
    rs.stream.color,
    1280,
    720,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    1280,
    720,
    rs.format.z16,
    30
)


profile = pipeline.start(config)


# 深度对齐RGB
align = rs.align(rs.stream.color)


# =====================
# 获取相机参数
# =====================

depth_sensor = profile.get_device().first_depth_sensor()

depth_scale = depth_sensor.get_depth_scale()


depth_profile = profile.get_stream(
    rs.stream.depth
)

intr = depth_profile.as_video_stream_profile().get_intrinsics()


camera_info = {

    "fx": intr.fx,
    "fy": intr.fy,
    "cx": intr.ppx,
    "cy": intr.ppy,

    "width": intr.width,
    "height": intr.height,

    "depth_scale": depth_scale
}


with open(
    os.path.join(save_dir,"camera.json"),
    "w"
) as f:

    json.dump(
        camera_info,
        f,
        indent=4
    )


print("开始采集")
print("按 s 保存")
print("按 q 退出")


index = 0


try:

    while True:


        frames = pipeline.wait_for_frames()


        # RGB-D对齐
        aligned_frames = align.process(frames)


        color_frame = aligned_frames.get_color_frame()

        depth_frame = aligned_frames.get_depth_frame()


        if not color_frame or not depth_frame:
            continue



        color = np.asanyarray(
            color_frame.get_data()
        )


        depth = np.asanyarray(
            depth_frame.get_data()
        )


        # 显示
        cv2.imshow(
            "RGB",
            color
        )


        depth_show = cv2.convertScaleAbs(
            depth,
            alpha=0.03
        )

        cv2.imshow(
            "Depth",
            depth_show
        )



        key = cv2.waitKey(1)



        # =====================
        # 保存
        # =====================

        if key == ord('s'):


            name = f"{index:06d}"


            # RGB
            cv2.imwrite(
                f"{rgb_dir}/{name}.png",
                color
            )


            # Depth
            cv2.imwrite(
                f"{depth_dir}/{name}.png",
                depth
            )


            # PointCloud
            pc = rs.pointcloud()

            points = pc.calculate(
                depth_frame
            )

            vertices = np.asanyarray(
                points.get_vertices()
            )


            xyz = np.asarray(vertices).view(
                np.float32
            ).reshape(-1,3)


            xyz = xyz[
                np.any(xyz!=0,axis=1)
            ]


            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(
                xyz
            )


            o3d.io.write_point_cloud(
                f"{pc_dir}/{name}.ply",
                pcd
            )


            print(
                "保存:",
                name
            )


            index += 1



        elif key == ord('q'):
            break


finally:

    pipeline.stop()

    cv2.destroyAllWindows()