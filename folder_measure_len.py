      
from pathlib import Path
import cv2
import numpy as np
from PySide2.QtCore import QPointF
from imgproc.camera_model import CameraModel
from imgproc.rectification import StereoRectify
from auto_match.aruco import ArucoDetector
from icecream import ic
import mrcal 
from imgproc.mrcal_rectify import load_cameramodel, rectify_image, unproject_to_3dcoords
import pandas as pd
import open3d as o3d
import os

folder_name = '072813' 
TARGET_FOLDER = Path().home() / 'DCIM' / folder_name 
TARGET_FOLDER2 = Path().home() / 'DCIM'
camera00 = "/mnt/e/stereovision_camera/calibration_images/20250728_new_6mm_lv/cv/camera0-0.cameramodel"
camera01 = "/mnt/e/stereovision_camera/calibration_images/20250728_new_6mm_lv/cv/camera0-1.cameramodel"


# 设置环境变量
os.environ['FOLDER_NAME'] = folder_name
csv_file_path = os.path.join(TARGET_FOLDER2, f'{folder_name}.csv')

# 添加额外的坐标对
# id_pairs = [{33: -1, 34: -1}, {33: -1, 35: -1}, {34: -1, 35: -1},{34: -1, 37: -1}, {35: -1, 37: -1}, {35: -1, 38: -1},{37: -1, 38: -1},
#             {46: -1, 47: -1}, {46: -1, 48: -1}, {47: -1, 48: -1},{47: -1, 51: -1}, {48: -1, 51: -1}, {48: -1, 56: -1},{51: -1, 56: -1},
#             {24: -1, 25: -1}, {24: -1, 26: -1}, {25: -1, 26: -1},{25: -1, 28: -1}, {26: -1, 28: -1}, {26: -1, 29: -1},{28: -1, 29: -1}]

# values = {(33, 34): 356.5, (33, 35): 752, (34, 35): 401 , (34, 37): 781, (35, 37): 380, (35, 38): 769, (37, 38): 389,
#           (46, 47): 355.5, (46, 48): 757, (47, 48): 401, (47, 51): 781, (48, 51): 380, (48, 56): 769.5, (51, 56): 389.5,
#           (24, 25): 357, (24, 26): 757, (25, 26): 401, (25, 28): 781, (26, 28): 380, (26, 29): 770.5, (28, 29): 391}


excel_file_path = TARGET_FOLDER2 / 'lengthsGMroom.xlsx'
df = pd.read_excel(excel_file_path)

id_pairs = []
ids_dists = {}

###################################################

# 读取全站仪测量的点坐标文件
data = pd.read_excel(excel_file_path)

# 获取"id"列的数值列表
arucoid = data['id'].tolist()

# 创建id_pairs列表
id_pairs = [{arucoid[i]: -1, arucoid[j]: -1} for i in range(len(arucoid)) for j in range(i + 1, len(arucoid))]

ids_dists = {}
# 获取每两个id之间的距离
for i in range(len(data)):
    point1 = np.array([data['n'][i], data['e'][i], data['z'][i]])
    for j in range(i + 1, len(data)):
        point2 = np.array([data['n'][j], data['e'][j], data['z'][j]])
        dist_2ids = np.linalg.norm(point1 - point2)
        ids_dists[(data['id'][i], data['id'][j])] = dist_2ids

###################################################

# # 遍历DataFrame中的每一行
# for index, row in df.iterrows():
#     # 将ID列的内容转换为字典
#     id_pair = {int(id): -1 for id in eval(row['id'])}
#     id_pairs.append(id_pair)
    
#     # 将len列的内容转换为字典
#     id_tuple = tuple(sorted(eval(row['id'])))
#     ids_dists[id_tuple] = row['len']

def numpy_to_qpointf(np_array):
    return QPointF(np_array[0], np_array[1])

def get_world_coord_Q(left: np.ndarray, right: np.ndarray, Q):
    """Compute world coordniate by the Q matrix
    
    img_coord_left:  segment endpoint coordinate on the  left image
    img_coord_right: segment endpoint coordinate on the right image
    """
    left = numpy_to_qpointf(left)
    right = numpy_to_qpointf(right)
    x, y = left.x(), left.y()
    d = left.x() - right.x()
    # print(x, y, d); exit(0)
    homg_coord = Q.dot(np.array([x, y, d, 1.0]))
    coord = homg_coord / homg_coord[3]
    # print(coord[:-1])
    return coord[:-1]


# determine the suffix of the file
suffix = list(TARGET_FOLDER.glob("A_*"))[0].suffix

# dataframe for saving
df = pd.DataFrame(columns=["id_pair", "length", "true_value", "error", "error_abs", "filename","id1","id2","point1_world","point2_world","coord1_left","coord2_left","coord1_right","coord2_right"])
# df = pd.DataFrame(columns=["id_pair", "length", "filename", "class"])

# iterate through the folder\
for left_img_file in TARGET_FOLDER.glob(f"A_*{suffix}"):
    id = left_img_file.stem[2:]
    right_img_file = TARGET_FOLDER / f"D_{id}{suffix}"
    # Now left_img_file and right_img_file examples:
    #   /test/A_1.jpg       /test/D_1.jpg
    # print the left and right image file
    ic(id)

    # Step 1: Load the images
    # left_img = cv2.imread(str(left_img_file))
    # right_img = cv2.imread(str(right_img_file))

    # Step 2: rectify the images using the imgproc.rectification module
    models = load_cameramodel(camera00, camera01)
    images_rectified, models_rectified = rectify_image(str(left_img_file), str(right_img_file), models)
    left_rect, right_rect = images_rectified

    # stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*5, blockSize=5)  # 可根据需要调整参数
            
    # disparity_map1 = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
    # cv2.imwrite("test.jpg",disparity_map1)

    # points_3d = cv2.reprojectImageTo3D(disparity_map1, Q)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_3d.reshape(-1, 3))
    # o3d.io.write_point_cloud("test.ply", pcd)

    # 显示视差图
    # cv2.imshow("Disparity Map 1", disparity_map1_visual)
    # cv2.imshow("Disparity Map 2", disparity_map2_visual)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # camera_model = CameraModel.load_model(TARGET_CM)
    # rectifier = StereoRectify(camera_model, None)
    # left_rect, right_rect = rectifier.rectify_image(
    #     left_img=left_img,
    #     right_img=right_img,
    # )
    # Q = rectifier.Q

    # Step 3: detect aruco marker pairs
    left_ad = ArucoDetector(left_rect)
    right_ad = ArucoDetector(right_rect)
    left_view_points = []
    right_view_points = []

    # expandable list of id_pairs


    for id_pair in id_pairs:
        try:
            left_result = left_ad.get_aruco_pairs(id_pair)
            right_result = right_ad.get_aruco_pairs(id_pair)
            if left_result is None or right_result is None:
                raise IndexError

    # Step 4: compute the world coordinate of each marker pairs
            point1_left = left_result[0]
            point1_right = right_result[0]
            d = point1_left[0] - point1_right[0]
            ic(point1_left)
            # unproject
            point1_world = unproject_to_3dcoords(point1_left, d, models, models_rectified)
            point2_left = left_result[1]
            point2_right = right_result[1]
            d = point2_left[0] - point2_right[0]
            # unproject
            point2_world = unproject_to_3dcoords(point2_left, d, models, models_rectified)


    # Step 5: compute the length of each marker pairs by l2 norm
            length = np.linalg.norm(point1_world - point2_world) * 1e3
            # print info about this id pair and its length
            print(f"\taruco id: {id_pair.keys()} length: {length}")
# #################################################################################
            id_tuple = tuple(id_pair.keys())  # 将字典的键转换为元组
            id1, id2 = id_tuple
            distance = ids_dists.get(id_tuple)
            error = (length - distance)
            error_abs = abs(error)

            df.loc[len(df)] = {                    
                    "id_pair": str(list(id_pair.keys())),
                    "length": length,
                    "true_value": distance,
                    "error": error,
                    "error_abs": error_abs,       
                    "id1": id1,
                    "id2": id2,
                    "point1_world": point1_world,
                    "point2_world": point2_world,
                    "filename":left_img_file.stem,
                    "coord1_left": point1_left,
                    "coord2_left": point2_left,  
                    "coord1_right": point1_right,
                    "coord2_right": point2_right,           
                }

        except IndexError:
            print(f"\tGet {id_pair.keys()} aruco failed!")
            continue


import ast

# 保存过滤后的数据为CSV文件
# csv_file_path = TARGET_FOLDER / "results.csv"
sorted_df = df.sort_values(by=["true_value"], ascending=[True])
sorted_df.to_csv(csv_file_path, index=False)

error_distrib_path = os.path.join(os.path.dirname(__file__), 'error_distrib.py')
os.system(f'python {error_distrib_path}')