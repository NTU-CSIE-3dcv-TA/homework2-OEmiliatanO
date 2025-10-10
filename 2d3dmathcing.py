from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
import os
from transform_cube import load_point_cloud, load_axes, get_transform_mat, create_colored_pointcube

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # solve PnP problem using OpenCV
    FLANN_INDEX_KDTREE = 1
    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_query, desc_model, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    pts_2d = np.array([kp_query[m.queryIdx] for m in good_matches], dtype=np.float32)
    pts_3d = np.array([kp_model[m.trainIdx] for m in good_matches], dtype=np.float32)

    flags=cv2.SOLVEPNP_ITERATIVE
    # flags=cv2.SOLVEPNP_EPNP
    # flags=cv2.SOLVEPNP_P3P
    # flags=cv2.SOLVEPNP_DLS
    # flags=cv2.SOLVEPNP_UPNP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, cameraMatrix, distCoeffs,
        flags=flags
    )

    if success:
        return success, rvec, tvec, inliers
    else:
        return False, None, None, None

def rotation_error(R1, R2):
    # calculate rotation error
    R1 = R.from_quat(R1)
    R2 = R.from_quat(R2)
    relative_R = R1.inv() * R2
    angle_rad = relative_R.magnitude()
    return angle_rad

def translation_error(t1, t2):
    # calculate translation error
    return np.sqrt(np.sum((t1-t2)**2, axis=1))

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    # visualize the camera pose
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    
    axes = load_axes()
    vis.add_geometry(axes)

    # cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    # cube_vertices = np.asarray(cube.vertices).copy()
    # vis.add_geometry(cube)

    def create_camera_frustum(scale=0.1, color=[1, 0, 0]):
        # camera center
        o = np.array([0, 0, 0])

        p1 = np.array([-0.5, -0.5, 1]) * scale
        p2 = np.array([ 0.5, -0.5, 1]) * scale
        p3 = np.array([ 0.5,  0.5, 1]) * scale
        p4 = np.array([-0.5,  0.5, 1]) * scale

        points = [o, p1, p2, p3, p4]
        lines = [
            [0,1],[0,2],[0,3],[0,4],
            [1,2],[2,3],[3,4],[4,1]
        ]
        colors = [color for _ in lines]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    camera_frustums = []
    trajectory_points = []

    for c2w in Camera2World_Transform_Matrixs:
        frustum = create_camera_frustum(scale=0.1)
        frustum.transform(c2w)
        camera_frustums.append(frustum)
        vis.add_geometry(frustum)

        trajectory_points.append(c2w[:3, 3].tolist())
    
    from scipy.spatial import distance_matrix
    dist_mat = distance_matrix(trajectory_points, trajectory_points)
    for i in range(0, len(trajectory_points)): dist_mat[i,i] = np.inf
    now = 5
    ordered_trajectory_points = []
    ordered_index = []
    dist_mat[:,now] = np.inf
    for i in range(0, len(trajectory_points)):
        ordered_trajectory_points.append(trajectory_points[now])
        ordered_index.append(now)
        nex = np.argmin(dist_mat[now])
        dist_mat[:,nex] = dist_mat[nex,now] = np.inf
        now = nex

    trajectory = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ordered_trajectory_points),
        lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(ordered_trajectory_points)-1)])
    )
    trajectory.colors = o3d.utility.Vector3dVector([[0,1,0] for _ in range(len(trajectory.lines))])
    vis.add_geometry(trajectory)

    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    vis.run()
    return ordered_index

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_LIST = [[n,i] for n,i in images_df[["NAME","IMAGE_ID"]].values if 'valid_' in n]
    IMAGE_NAME_LIST, IMAGE_ID_LIST = [*zip(*IMAGE_LIST)]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST, desc="Solving camera pose"):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        success, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        if not success:
            print(f"WARNING: Cannot solve the PNP problem (the {idx}-th image). Skip this image.")
            continue
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        tvec = tvec.reshape(1,3)
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq_gt, rotq)
        t_error = translation_error(tvec_gt, tvec)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # calculate median of relative rotation angle differences and translation differences and print them
    median_idx = len(rotation_error_list) // 2
    median_rotation_err = sorted(rotation_error_list)[median_idx]
    median_translation_err = sorted(translation_error_list)[median_idx]
    print(f"{median_rotation_err=}")
    print(f"{median_translation_err=}")

    # result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # calculate camera pose in world coordinate system
        R, _ = cv2.Rodrigues(r)
        t = t.reshape(3, 1)
        R_c2w = R.T
        C = -R.T @ t
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = C.flatten()
        Camera2World_Transform_Matrixs.append(c2w)
    # Q2-1
    ordered_index = visualization(Camera2World_Transform_Matrixs, points3D_df) 
    
    # Q2-2
    from pathlib import Path
    photo_dir = Path(f"data/frames/")
    photo_paths = [photo_dir / Path(IMAGE_NAME_LIST[i]) for i in ordered_index]
    output_dir = Path("outputs/")
    os.makedirs(output_dir, exist_ok=True)

    # Reorder the camera pose
    Camera2World_Transform_Matrixs = np.array(Camera2World_Transform_Matrixs)
    Camera2World_Transform_Matrixs = Camera2World_Transform_Matrixs[np.array(ordered_index)]

    # Create cube
    cube = create_colored_pointcube(shift=np.array([0, 0, 1]), size=0.5, points_per_face=64, sphere_radius=0.01)

    width, height = 1080, 1920
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 1868.27, 1869.18, 540, 960)
    
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0, 0, 0, 0])
    scene.scene.set_sun_light(np.array([[0], [0], [0]]), np.array([[1], [1], [1]]), 1)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    cube_mat = o3d.visualization.rendering.MaterialRecord()
    cube_mat.shader = "defaultLit"
    # cube_mat.shader = "defaultUnlit"
    # cube_mat.base_color = [1, 1, 1, 1]
    
    pcd = load_point_cloud(points3D_df)
    scene.add_geometry("pcd", pcd, mat)
    scene.add_geometry("cube", cube, cube_mat)
    
    DEPTH_EPS = 1e-8
    output_images_path = []
    video_writer = cv2.VideoWriter('AR_camera_trajectory.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
    for i, (c2w, photo_path) in tqdm(enumerate(zip(Camera2World_Transform_Matrixs, photo_paths)), total=len(photo_paths), desc="Rendering pictures with a virtual cube"):
        w2c = np.linalg.inv(c2w)
        renderer.setup_camera(intrinsic, w2c)

        scene.remove_geometry("cube")
        renderer.setup_camera(intrinsic, w2c)

        depth_real_o3d = renderer.render_to_depth_image(z_in_view_space=True)
        depth_real = np.asarray(depth_real_o3d)

        scene.remove_geometry("pcd")
        scene.add_geometry("cube", cube, cube_mat)
        renderer.setup_camera(intrinsic, w2c)

        img_cube_o3d = renderer.render_to_image()                # RGBA image (Open3D image)
        depth_cube_o3d = renderer.render_to_depth_image(True)    # float depth in camera-space
        img_cube = np.asarray(img_cube_o3d)
        depth_cube = np.asarray(depth_cube_o3d)

        real_img = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        real_img = cv2.resize(real_img, (width, height))

        mask_cube_pixels = depth_cube > 0.0
        mask_real_pixels = depth_real > 0.0
        visible_cube = mask_cube_pixels & (~mask_real_pixels | (depth_cube < depth_real - DEPTH_EPS))

        cube_rgb = img_cube[:, :, :3].astype(np.float32)
        result = real_img.astype(np.float32)

        for c in range(3):
            result[..., c] = np.where(
                visible_cube,
                cube_rgb[..., c],
                result[..., c]
            )

        # output the picture
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

        out_path = str(output_dir / Path(f"augmented_{i:03d}.png"))
        output_images_path.append(out_path)
        cv2.imwrite(out_path, result_bgr)
        video_writer.write(result_bgr)

        scene.remove_geometry("cube")
        scene.add_geometry("pcd", pcd, mat)
        scene.add_geometry("cube", cube, cube_mat)
    
    # make video
    video_writer.release()
    print("Outputs the video at ./AR_camera_trajectory.mp4")