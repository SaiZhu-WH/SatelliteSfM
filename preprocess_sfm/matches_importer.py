import os
import json
import numpy as np
import sqlite3
from collections import defaultdict
from tqdm import tqdm

# 导入同目录下的数据库工具
from .colmap import database
from .colmap.database import COLMAPDatabase

# ==========================================
# 【关键修复】: 动态修正 database.py 中的非标准列名
# 将 data_500 替换为 data，以匹配标准 COLMAP 程序的要求
# ==========================================
def patch_database_schema():
    # 检查是否需要 Patch (如果已经是 data 就不改了)
    if "data_500" in database.CREATE_KEYPOINTS_TABLE:
        print("[Matches Importer] Patching database schema: data_500 -> data")
        database.CREATE_DESCRIPTORS_TABLE = database.CREATE_DESCRIPTORS_TABLE.replace("data_500", "data")
        database.CREATE_KEYPOINTS_TABLE = database.CREATE_KEYPOINTS_TABLE.replace("data_500", "data")
        database.CREATE_MATCHES_TABLE = database.CREATE_MATCHES_TABLE.replace("data_500", "data")
        database.CREATE_TWO_VIEW_GEOMETRIES_TABLE = database.CREATE_TWO_VIEW_GEOMETRIES_TABLE.replace("data_500", "data")
        
        # 重新生成 CREATE_ALL 语句
        database.CREATE_ALL = "; ".join([
            database.CREATE_CAMERAS_TABLE,
            database.CREATE_IMAGES_TABLE,
            database.CREATE_KEYPOINTS_TABLE,
            database.CREATE_DESCRIPTORS_TABLE,
            database.CREATE_MATCHES_TABLE,
            database.CREATE_TWO_VIEW_GEOMETRIES_TABLE,
            database.CREATE_NAME_INDEX
        ])

# 在脚本加载时立即执行补丁
patch_database_schema()

def import_matches_json(output_folder, json_path):
    """
    读取 matches.json 并生成 database.db
    """
    img_dir = os.path.join(output_folder, 'images')
    camera_dir = os.path.join(output_folder, 'cameras')
    db_path = os.path.join(output_folder, 'database.db')
    
    print(f"\n[Matches Importer] Importing from: {json_path}")
    print(f"[Matches Importer] Target Database: {db_path}")

    if not os.path.exists(json_path):
        print(f"Error: matches.json not found at {json_path}")
        return False

    # 1. 读取 JSON 数据
    with open(json_path, 'r') as f:
        matches_data = json.load(f)
        
    if not matches_data:
        print("Error: Empty matches file.")
        return False

    # 2. 初始化数据库
    # 如果数据库已存在，先删除，确保重新建表时使用新的 Schema
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # 连接数据库
    db = COLMAPDatabase.connect(db_path)
    # 强制使用我们 patch 过的 create_tables
    db.create_tables = lambda: db.executescript(database.CREATE_ALL)
    db.create_tables()

    # 3. 整理特征点和匹配关系
    all_keypoints = defaultdict(list)
    all_matches = {} 
    
    valid_images = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()

    print("[Matches Importer] Parsing matches data...")
    for pair in tqdm(matches_data):
        img0_name = pair['image0']
        img1_name = pair['image1']
        
        # 简单后缀修复逻辑
        if img0_name not in valid_images:
            if img0_name.replace('.tif', '.png') in valid_images:
                img0_name = img0_name.replace('.tif', '.png')
        if img1_name not in valid_images:
            if img1_name.replace('.tif', '.png') in valid_images:
                img1_name = img1_name.replace('.tif', '.png')
                
        start_idx0 = len(all_keypoints[img0_name])
        start_idx1 = len(all_keypoints[img1_name])
        
        m0 = pair['matches0']
        m1 = pair['matches1']
        
        all_keypoints[img0_name].extend(m0)
        all_keypoints[img1_name].extend(m1)
        
        num_matches = len(m0)
        current_indices = np.stack([
            np.arange(start_idx0, start_idx0 + num_matches),
            np.arange(start_idx1, start_idx1 + num_matches)
        ], axis=1)
        
        all_matches[(img0_name, img1_name)] = current_indices

    # 4. 写入相机和图片
    print("[Matches Importer] Writing cameras and images...")
    image_name_to_id = {}
    
    all_image_names = sorted(all_keypoints.keys())
    
    for img_name in all_image_names:
        cam_json = os.path.join(camera_dir, os.path.splitext(img_name)[0] + '.json')
        
        if os.path.exists(cam_json):
            with open(cam_json) as f:
                cam_data = json.load(f)
            
            # 适配 img_size 字段
            if 'img_size' in cam_data:
                w, h = int(cam_data['img_size'][0]), int(cam_data['img_size'][1])
            elif 'width' in cam_data and 'height' in cam_data:
                w, h = int(cam_data['width']), int(cam_data['height'])
            else:
                print(f"Warning: Could not find size for {img_name}, using default.")
                w, h = 2048, 2048

            # 适配扁平化的 K 矩阵
            K = cam_data['K']
            if isinstance(K[0], list): # 二维列表
                fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            else: # 一维列表 (16个元素)
                # K[0]=fx, K[5]=fy, K[2]=cx, K[6]=cy (对应 4x4 矩阵的对角线和偏移)
                fx, fy, cx, cy = K[0], K[5], K[2], K[6]

            params = np.array([fx, fy, cx, cy])
            cam_id = db.add_camera(1, w, h, params) # 1 = PINHOLE
        else:
            print(f"Warning: Camera file not found for {img_name}")
            cam_id = db.add_camera(1, 1000, 1000, np.array([1000., 1000., 500., 500.]))
            
        img_id = db.add_image(img_name, cam_id)
        image_name_to_id[img_name] = img_id

    # 5. 写入特征点
    print("[Matches Importer] Writing keypoints...")
    for img_name, kpts in all_keypoints.items():
        img_id = image_name_to_id[img_name]
        if len(kpts) == 0:
            kpts_arr = np.zeros((0, 2), dtype=np.float32)
        else:
            kpts_arr = np.array(kpts, dtype=np.float32)
        db.add_keypoints(img_id, kpts_arr)

    # 6. 写入匹配
    print("[Matches Importer] Writing matches...")
    count = 0
    for (name0, name1), indices in all_matches.items():
        if name0 in image_name_to_id and name1 in image_name_to_id:
            id0 = image_name_to_id[name0]
            id1 = image_name_to_id[name1]
            if len(indices) > 0:
                # 写入 Verified Matches (two_view_geometries)
                db.add_two_view_geometry(id0, id1, indices, config=2)
                count += 1
    
    db.commit()
    db.close()
    print(f"[Matches Importer] Success! Database ready with {count} matched pairs.")
    return True