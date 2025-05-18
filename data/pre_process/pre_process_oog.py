import numpy as np
import torch
import os
import json
import copy
from pycocotools.coco import COCO
from common.utils.mano import MANO  #Replace it with your own path
mano = MANO()
import matplotlib.pyplot as plt
from process_util import (R_T_to_matrix,rvec_tvec_to_matrix,rvec_tvec_to_matrix,
                          matrix_to_rvec_tvec,matrix_to_R_T,rotate_translate_point_cloud,
                          is_point_inside_cube,compute_bounding_box_corners,load_yaml_config,
                          get_bbox,process_bbox)

'''
We suggest that you obtain the grasping information, object name information, etc. in advance to quickly get the json of the text prompt.
This might require certain modifications to the code in order to obtain the json information. 
We have provided a demo of the relevant jsons so that the sample of obtaining the text prompt can be run.
'''
joint_mapping = {
    0: "Wrist",
    1: "MCP of the thumb", 2: "PIP of the thumb", 3: "DIP of the thumb", 4: "TIP of the thumb",
    5: "MCP of the index finger", 6: "PIP of the index finger", 7: "DIP of the index finger",
    8: "TIP of the index finger",
    9: "MCP of the middle finger", 10: "PIP of the middle finger", 11: "DIP of the middle finger",
    12: "TIP of the middle finger",
    13: "MCP of the ring finger", 14: "PIP of the ring finger", 15: "DIP of the ring finger",
    16: "TIP of the ring finger",
    17: "MCP of the pinky", 18: "PIP of the pinky", 19: "DIP of the pinky", 20: "TIP of the pinky"
}

ho3d_name_mapping = {
            'ABF10': 'bleach cleanser', 'ABF11': 'bleach cleanser', 'ABF12': 'bleach cleanser',
            'ABF13': 'bleach cleanser', 'ABF14': 'bleach cleanser',
            'BB10': 'banana', 'BB11': 'banana', 'BB12': 'banana', 'BB13': 'banana', 'BB14': 'banana',
            'GPMF10': 'potted meat can', 'GPMF11': 'potted meat can', 'GPMF12': 'potted meat can',
            'GPMF13': 'potted meat can', 'GPMF14': 'potted meat can',
            'GSF10': 'scissors', 'GSF11': 'scissors', 'GSF12': 'scissors', 'GSF13': 'scissors', 'GSF14': 'scissors',
            'MC1': 'cracker box', 'MC2': 'cracker box', 'MC3': 'cracker box', 'MC4': 'cracker box',
            'MC5': 'cracker box', 'MC6': 'cracker box',
            'MDF10': 'power drill', 'MDF11': 'power drill', 'MDF12': 'power drill', 'MDF13': 'power drill',
            'MDF14': 'power drill',
            'ND2': 'power drill',
            'SB10': 'bleach cleanser', 'SB12': 'bleach cleanser', 'SB14': 'bleach cleanser',
            'ShSu10': 'sugar box', 'ShSu12': 'sugar box', 'ShSu13': 'sugar box', 'ShSu14': 'sugar box',
            'SiBF10': 'banana', 'SiBF11': 'banana', 'SiBF12': 'banana', 'SiBF13': 'banana', 'SiBF14': 'banana',
            'SiS1': 'sugar box',
            'SM2': 'mustard bottle', 'SM3': 'mustard bottle', 'SM4': 'mustard bottle', 'SM5': 'mustard bottle',
            'SMu1': 'mug', 'SMu41': 'mug', 'SMu40': 'mug', 'SMu42': 'mug',
            'SS1': 'sugar box', 'SS2': 'sugar box', 'SS3': 'sugar box',
            # test
            'AP10':'pitcher base','AP11':'pitcher base','AP12':'pitcher base','AP13':'pitcher base','AP14':'pitcher base',
            'MCM10':'potted meat can','MCM11':'potted meat can','MCM12':'potted meat can','MCM13':'potted meat can','MCM14':'potted meat can',
            'SB11':'bleach cleanser','SB13':'bleach cleanser',
            'SM1':'mustard bottle'
        }

def if_grab_HO3D(data):
    '''
    Pre-grasping and grasping judgments on the ho3d training set.
    The data can be obtained from the official ho3d annotations JSON files.
    '''
    hand_joints_3d = data['handJoints3D']
    obj_trans = data['objTrans']
    obj_rot = data['objRot']
    obj_corners_3d = data['objCorners3D']
    obj_to_cam_matrix = rvec_tvec_to_matrix(obj_rot, obj_trans)

    cam_to_obj_matrix = np.linalg.inv(obj_to_cam_matrix)

    hand_joints_3d_objCoordinate = rotate_translate_point_cloud(hand_joints_3d, cam_to_obj_matrix)

    obj_corners_3d_objCoordinate = data['objCorners3DRest'].reshape(-1, 3)

    points_inside = np.array([
        is_point_inside_cube(joint, obj_corners_3d_objCoordinate)
        for joint in hand_joints_3d_objCoordinate
    ])

    thumb_indices = [1, 2, 3, 4]
    other_indices = [i for i in range(5, 21)]

    has_thumb_contact = np.any(points_inside[thumb_indices])
    has_other_contact = np.any(points_inside[other_indices])

    return has_thumb_contact and has_other_contact

def load_dexycb_data(data_split):
    """load DEX_YCB"""
    root_dir = "/home/dataset/DEX_YCB/"
    annot_path = os.path.join(root_dir, "annotations")
    db = COCO('ycb_filtered.json') #Your own path

    datalist = []
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann["image_id"]
        img = db.loadImgs(image_id)[0]

        data = {
            "image_id": image_id,
            "img_path": os.path.join(root_dir, img["file_name"]),
            "img_shape": (img["height"], img["width"]),
            "joints_coord_cam": np.array(ann["joints_coord_cam"], dtype=np.float32),
            "hand_type": ann["hand_type"],
            "cam_param": {k: np.array(v, dtype=np.float32) for k, v in ann["cam_param"].items()},
            "mano_pose": np.array(ann["mano_param"]["pose"], dtype=np.float32),
            "mano_shape": np.array(ann["mano_param"]["shape"], dtype=np.float32)
        }

        joints_img = np.array(ann["joints_img"], dtype=np.float32)
        bbox = get_bbox(joints_img[:, :2], np.ones_like(joints_img[:, 0]), 1.5)
        data["bbox"] = process_bbox(bbox, img["width"], img["height"], 1.0) if data_split == "train" else \
            np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

        datalist.append(data)
    return datalist

def process_grasp_detection(data, meta_path):
    """grasping detection"""
    meta = load_yaml_config(meta_path)
    ycb_ids = meta['ycb_ids']
    grasp_idx = meta['ycb_grasp_ind']
    obj_id = ycb_ids[grasp_idx]

    model_dir = None #Your own path
    obj_folder = sorted(os.listdir(model_dir))[obj_id - 1]


    obj_path = os.path.join(model_dir, obj_folder, "points.xyz")
    obj_cloud = np.loadtxt(obj_path)
    obj_bbox = compute_bounding_box_corners(obj_cloud)

    label_path = data["img_path"].replace("color", "labels").replace("jpg", "npz")
    pose_data = np.load(label_path)
    obj_pose = pose_data["pose_y"][grasp_idx]
    obj_pose = np.vstack([obj_pose, [0, 0, 0, 1]])

    # Coordinate transformation
    joints_cam = data["joints_coord_cam"]
    homogeneous_joints = np.hstack([joints_cam, np.ones((len(joints_cam), 1))])
    joints_obj = (np.linalg.inv(obj_pose) @ homogeneous_joints.T).T[:, :3]

    # Contact detection
    thumb_indices = [1, 2, 3, 4]
    other_indices = list(range(5, 21))

    thumb_inside = any(is_point_inside_cube(joints_obj[i], obj_bbox) for i in thumb_indices)
    other_inside = any(is_point_inside_cube(joints_obj[i], obj_bbox) for i in other_indices)

    obj_name = ' '.join([p for p in obj_folder.split('_') if not p.isdigit()]).lower()
    return (thumb_inside and other_inside), obj_name

def visualize_grasp(data, meta_path, save_path):
    """visualize"""
    meta = load_yaml_config(meta_path)
    ycb_ids = meta['ycb_ids']
    grasp_idx = meta['ycb_grasp_ind']
    obj_id = ycb_ids[grasp_idx]

    model_dir = "/home/dataset/DEX_YCB/models/"
    obj_folder = sorted(os.listdir(model_dir))[obj_id - 1]

    obj_path = os.path.join(model_dir, obj_folder, "points.xyz")
    obj_cloud = np.loadtxt(obj_path)

    label_path = data["img_path"].replace("color", "labels").replace("jpg", "npz")
    pose_data = np.load(label_path)
    obj_pose = pose_data["pose_y"][grasp_idx]
    obj_pose = np.vstack([obj_pose, [0, 0, 0, 1]])

    joints_cam = data["joints_coord_cam"]
    homogeneous_joints = np.hstack([joints_cam, np.ones((len(joints_cam), 1))])
    joints_obj = (np.linalg.inv(obj_pose) @ homogeneous_joints.T).T[:, :3]


    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # object points clouds
    ax.scatter(obj_cloud[:, 0], obj_cloud[:, 1], obj_cloud[:, 2],
               c='gray', s=1, alpha=0.3, label='Object Point Cloud')

    # joints
    thumb = joints_obj[[1, 2, 3, 4]]
    other = joints_obj[5:]
    ax.scatter(thumb[:, 0], thumb[:, 1], thumb[:, 2],
               c='red', s=50, marker='o', label='Thumb Joints')
    ax.scatter(other[:, 0], other[:, 1], other[:, 2],
               c='blue', s=50, marker='^', label='Other Joints')

    # bounding box
    bbox = compute_bounding_box_corners(obj_cloud)
    corners = np.array([
        [bbox[0, 0], bbox[0, 1], bbox[0, 2]],  # point 0
        [bbox[1, 0], bbox[0, 1], bbox[0, 2]],  #  1
        [bbox[1, 0], bbox[1, 1], bbox[0, 2]],  #  2
        [bbox[0, 0], bbox[1, 1], bbox[0, 2]],  #  3
        [bbox[0, 0], bbox[0, 1], bbox[1, 2]],  #  4
        [bbox[1, 0], bbox[0, 1], bbox[1, 2]],  #  5
        [bbox[1, 0], bbox[1, 1], bbox[1, 2]],  #  6
        [bbox[0, 0], bbox[1, 1], bbox[1, 2]]   #  7
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical
    ]

    for i, edge in enumerate(edges):
        label = 'Bounding Box' if i == 0 else None
        ax.plot3D(*corners[edge].T,
                  color='limegreen',
                  linestyle='--',
                  linewidth=1.5,
                  alpha=0.8,
                  label=label)

    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.view_init(elev=20, azim=45)  # 设置视角
    plt.title(f'Grasp Visualization - {obj_folder}', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
def process_official_version(data_split, output_path, start_idx=None, end_idx=None):
    """Main processing flow (visual call added)"""
    datalist = load_dexycb_data(data_split)
    total = len(datalist)
    start = max(0, start_idx) if start_idx else 0
    end = min(end_idx, total) if end_idx else total
    processed = datalist[start:end]

    results = {}
    for idx, data in enumerate(processed, start=1):
        try:
            img_dir = os.path.dirname(data["img_path"])
            meta_path = os.path.join(os.path.dirname(img_dir), "meta.yml")

            is_grasp, obj_name = process_grasp_detection(data, meta_path)

            if (start + idx) % 10 == 0:
                vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"sample_{start + idx}.png")
                visualize_grasp(data, meta_path, vis_path)
            # load occ files
            base_dir = os.path.dirname(data["img_path"])
            frame_num = os.path.basename(data["img_path"]).split("_")[-1].replace(".jpg", "")

            def load_occ(suffix):
                occ_path = os.path.join(base_dir, f"color_{frame_num}_{suffix}.json")
                with open(occ_path) as f:
                    return json.load(f)
            # result records
            rel_path = "/".join(data["img_path"].split("/")[4:])
            results[rel_path] = {
                "grasp": int(is_grasp),
                "obj_name": obj_name,
                "self_occ":load_occ("self_occ"),
                "obj_occ":load_occ("object_occ"),
            }

        except Exception as e:
            print(f"Error processing sample {start + idx}: {str(e)}")
            continue

    # save relults
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=0)
    print(f"Successfully processed {len(results)} samples. Saved to {output_path}")


############################################################################################################
# Run code below (process_official_version) to obtain the relevant information of the DexYCB dataset:
# 1.the name of object,
# 2.the grasp status,
# 3.self-occlusion and object-occlusion labels
# get the Object name, Occlusion labels, and Grasp status(OOG)
process_official_version(
    data_split="train", # train or test
    output_path="YCB_Train_OGG.json", # Your own output path
    start_idx=None,
    end_idx=None
)
# Modify the path and mode according to your requirements

#############################################################################################################

def joints_deal_YCB(object_obscured_data, self_obscured_data):

    both_vis_joints = [int(key) for key in self_obscured_data if
                       self_obscured_data[key] == 0 and object_obscured_data[key] == 0]
    both_vis_joints_num = len(both_vis_joints)
    return self_obscured_data, object_obscured_data, both_vis_joints, both_vis_joints_num
def generate_sentence_YCB(object_obscured_joints, self_obscured_joints, obscured_by):
    if not object_obscured_joints and not self_obscured_joints:
        return "All fingers are visible."
    sentence = ""
    both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints)
    self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]
    if object_obscured_joints:
        if len(object_obscured_joints) == 1:
            sentence += "The {} is obscured by the{}, ".format(object_obscured_joints[0], obscured_by)
        else:
            obscured_joints_str = ", ".join(object_obscured_joints)
            sentence += "{} are obscured by the {}, ".format(obscured_joints_str, obscured_by)
        if self_obscured_joints:
            if len(self_obscured_joints) == 1:
                sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
            else:
                self_obscured_joints_str = ", ".join(self_obscured_joints)
                sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
    else:
        if self_obscured_joints:
            if len(self_obscured_joints) == 1:
                sentence += "The {} is obscured by other fingers".format(self_obscured_joints[0])
            else:
                self_obscured_joints_str = ", ".join(self_obscured_joints)
                sentence += "The {} are obscured by other fingers".format(self_obscured_joints_str)
    sentence = sentence.rstrip(', ')
    if sentence:
        sentence += ", and other fingers are visible."
    # if not sentence:
    #     sentence = "All fingers are visible."

    return sentence
def generate_local_text_YCB(object_occlusion_json, self_occlusion_json, obj_name):
    object_obscured_joints = [joint_mapping[int(i)] for i, obscured in object_occlusion_json.items() if
                              str(obscured) != '0']
    self_obscured_joints = [joint_mapping[int(i)] for i, obscured in self_occlusion_json.items() if
                            str(obscured) == '1']
    local_text = generate_sentence_YCB(object_obscured_joints, self_obscured_joints, obj_name)
    return local_text

def joints_deal_HO3D(object_obscured_joints,self_obscured_joints):

    both_vis_joints = [int(key) for key in self_obscured_joints if self_obscured_joints[key] == 0 and object_obscured_joints[key] == 0]
    both_vis_joints_num = len(both_vis_joints)
    return self_obscured_joints,object_obscured_joints,both_vis_joints,both_vis_joints_num

def generate_sentence_HO3D(object_obscured_joints, self_obscured_joints, obscured_by):
    if not object_obscured_joints and not self_obscured_joints:
        return "All fingers are visible."
    sentence = ""
    both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints)

    self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]

    if object_obscured_joints:
        if len(object_obscured_joints) == 1:
            sentence += "The {} is obscured by the {}, ".format(object_obscured_joints[0], obscured_by)
        else:
            obscured_joints_str = ", ".join(object_obscured_joints)
            sentence += "{} are obscured by the {}, ".format(obscured_joints_str, obscured_by)
        if self_obscured_joints:
            if len(self_obscured_joints) == 1:
                sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
            else:
                self_obscured_joints_str = ", ".join(self_obscured_joints)
                sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
    else:

        if self_obscured_joints:
            if len(self_obscured_joints) == 1:
                sentence += "The {} is obscured by other fingers".format(self_obscured_joints[0])
            else:
                self_obscured_joints_str = ", ".join(self_obscured_joints)
                sentence += "The {} are obscured by other fingers".format(self_obscured_joints_str)


    sentence = sentence.rstrip(', ')
    if sentence:
        sentence += ", and other fingers are visible."

    # if not sentence:
    #     sentence = "All fingers are visible."

    return sentence

def generate_local_text_HO3D(object_occlusion_json, self_occlusion_json,obj_name):

    object_obscured_joints = [
        joint_mapping[int(i)] for i, obscured in object_occlusion_json.items() if str(obscured) != '0'
    ]
    self_obscured_joints = [
        joint_mapping[int(i)] for i, obscured in self_occlusion_json.items() if str(obscured) == '1'
    ]
    local_text = generate_sentence_HO3D(object_obscured_joints, self_obscured_joints, obj_name)
    return local_text


