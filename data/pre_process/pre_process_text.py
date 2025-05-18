import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import numpy as np
import pickle
import cv2
from random import choice
# from manotorch.manolayer import ManoLayer, MANOOutput
from Long_CLIP_main.model import longclip
import torch.nn as nn
from torch.nn import functional as F
import json
import os.path as osp
from tqdm import tqdm
import re
import clip
import torch
from pycocotools.coco import COCO
from pre_process_oog import (joints_deal_YCB, generate_local_text_YCB,
                             joints_deal_HO3D, generate_local_text_HO3D)
# # #

# layer = nn.Sequential(
#     nn.Conv2d(1024, 512, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
# )
# input_tensor = torch.randn(16, 1024, 8, 8)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)


# #
# # import os
# # import torch
# # import numpy as np
# # import pickle
# # import cv2
# # from common.utils.mano import MANO
# # from scipy.spatial.transform import Rotation as R
# # mano = MANO()
# # from random import choice
# # import json
# # # 四个转换函数
# # def R_T_to_matrix(R, T):
# #     transform_matrix = np.eye(4)
# #     transform_matrix[0:3, 0:3] = R
# #     transform_matrix[0:3, 3] = T
# #     return transform_matrix
# #
# # def rvec_tvec_to_matrix(rvec, tvec):
# #     transform_matrix = np.eye(4)
# #     R, _ = cv2.Rodrigues(rvec)
# #     T = tvec.reshape((3,))
# #     transform_matrix[0:3, 0:3] = R
# #     transform_matrix[0:3, 3] = T
# #     return transform_matrix
# #
# # def matrix_to_rvec_tvec(transform_matrix):
# #     R = transform_matrix[0:3, 0:3]
# #     rvec, _ = cv2.Rodrigues(R)
# #     tvec = transform_matrix[0:3, 3]
# #     return rvec, tvec
# #
# # def matrix_to_R_T(transform_matrix):
# #     R = transform_matrix[0:3, 0:3]
# #     T = transform_matrix[0:3, 3]
# #     return R, T
# #
# # def rotate_translate_point_cloud(point_cloud, trans_matrix):
# #     homogeneous_coordinates = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
# #     new_coordinates = np.dot(trans_matrix, homogeneous_coordinates.T).T[:,:3]
# #     return new_coordinates
# #
# # # 定义一个函数来检查点是否在多边形内部（这里简化为立方体内）
# # def is_point_inside_cube(point, cube_corners):
# #     # 立方体的8个角点，我们需要检查点是否在这些角点构成的立方体内
# #     min_x = np.min(cube_corners[:, 0])
# #     max_x = np.max(cube_corners[:, 0])
# #     min_y = np.min(cube_corners[:, 1])
# #     max_y = np.max(cube_corners[:, 1])
# #     min_z = np.min(cube_corners[:, 2])
# #     max_z = np.max(cube_corners[:, 2])
# #
# #     return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y and min_z <= point[2] <= max_z
# #
# # # 检查符号是否相反
# # def are_signs_opposite(points):
# #     points_max_signs = np.sign(points.max(axis=0))
# #     points_min_signs = np.sign(points.min(axis=0))
# #     if points_max_signs[0]*points_min_signs[0]<0 or points_max_signs[1]*points_min_signs[1]<0 or points_max_signs[2]*points_min_signs[2]<0:
# #         return True
# #     else:
# #         return False
# #
# # def get_pkl_path(image_path):
# #     # 获取图片文件名（不包含扩展名）
# #     image_filename = os.path.splitext(os.path.basename(image_path))[0]
# #
# #     # 获取图片所在的目录路径
# #     image_dir = os.path.dirname(os.path.abspath(image_path))
# #
# #     # 构建meta目录的路径
# #     meta_dir = os.path.normpath(os.path.join(image_dir, '..', 'meta'))
# #
# #     # 构建.pkl文件的完整路径
# #     pkl_filename = f'{image_filename}.pkl'
# #     pkl_path = os.path.join(meta_dir, pkl_filename)
# #
# #     return pkl_path
# #
# # # 读取JSON文件并生成句子的函数
# #
# # # def generate_focal_text(object_occlusion_json, self_occlusion_json):
# # #     # 提取被物体遮挡和被手部自身遮挡的关节点名称
# # #     object_obscured_joints = [
# # #         joint_mapping[int(i)] for i, obscured in object_occlusion_json.items() if str(obscured) != '0'
# # #     ]
# # #     self_obscured_joints = [
# # #         joint_mapping[int(i)] for i, obscured in self_occlusion_json.items() if str(obscured) == '1'
# # #     ]
# # #     # both_obscured_joints = {
# # #     #     joint: object_occlusion_json[joint]
# # #     #     for joint in self_occlusion_json
# # #     #     if self_occlusion_json[joint] != 0 and object_occlusion_json[joint] != 0
# # #     # }
# # #     mapped_objects = {}
# # #     for key, value in object_occlusion_json.items():
# # #         # 检查value是否在obj_name_mapping中
# # #         if value in obj_name_mapping:
# # #             mapped_objects[key] = obj_name_mapping[value]
# # #         # else:
# # #         #     # 如果value不在obj_name_mapping中，可以选择忽略或记录错误信息
# # #         #     print(f"Warning: Key {key} maps to {value}, which is not in obj_name_mapping.")
# # #             # obj_name_dict = [
# # #     #     obj_name_mapping[int(i)] for i, obscured in object_occlusion_json.items() if str(obscured) != '0'
# # #     # ]
# # #     # 生成描述句子
# # #     focal_text = generate_sentence(object_obscured_joints, self_obscured_joints, mapped_objects)
# # #
# # #     return focal_text
# # # # def generate_sentence(object_obscured_joints, self_obscured_joints, obscured_by):
# # # #     # 初始化句子
# # # #     sentence = ""
# # # #     both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints) # 遮挡信息只有1的
# # # #
# # # #     self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]
# # # #     # 如果存在物体遮挡
# # # #     if object_obscured_joints:
# # # #         if len(object_obscured_joints) == 1:
# # # #             sentence += "The {} is obscured by {}, ".format(object_obscured_joints[0], obscured_by)
# # # #         else:
# # # #             obscured_joints_str = ", ".join(object_obscured_joints)
# # # #             sentence += "{} are obscured by {}, ".format(obscured_joints_str, obscured_by)
# # # #
# # # #     # 如果存在自身遮挡
# # # #     if self_obscured_joints:
# # # #         if len(self_obscured_joints) == 1:
# # # #             sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
# # # #         else:
# # # #             self_obscured_joints_str = ", ".join(self_obscured_joints)
# # # #             sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
# # # #
# # # #     # 清除句子尾部多余的逗号或空格
# # # #     sentence = sentence.rstrip(', ')
# # # #
# # # #     # 如果句子为空，则表示没有遮挡，添加 "All fingers are visible."
# # # #     if not sentence:
# # # #         sentence = "All fingers are visible."
# # # #
# # # #     # 添加 "and other fingers are visible."
# # # #     sentence += ", and other fingers are visible."
# # # #
# # # #     return sentence
# # # def generate_sentence(object_obscured_joints, self_obscured_joints, obscured_by_dict):
# # #     # 初始化句子
# # #     sentence = ""
# # #     both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints)
# # #
# # #     self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]
# # #     # 假设 self_obscured_joints 是一个字典，其键是关节点，值是是否被自身遮挡（True 或 False）
# # #     if object_obscured_joints:
# # #         # 首先，根据 obscured_by_dict，找出被不同物体遮挡的关节点
# # #         for obj, joints in obscured_by_dict.items():
# # #             if len(joints) == 1:
# # #                 sentence += "The {} is obscured by {}, ".format(object_obscured_joints[0], joints)
# # #             else:
# # #                 sentence += "{} are obscured by {}, ".format(object_obscured_joints[0], joints)
# # #
# # #         # 处理自身遮挡的关节点
# # #         # self_obscured_joints = ", ".join(self_obscured_joints)
# # #     if self_obscured_joints:
# # #         if len(self_obscured_joints) == 1:
# # #             sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
# # #         else:
# # #             self_obscured_joints_str = ", ".join(self_obscured_joints)
# # #             sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
# # #
# # #     # 清除句子尾部多余的逗号或空格
# # #     sentence = sentence.rstrip(', ')
# # #
# # #     # 如果句子为空，则表示没有遮挡
# # #     if not sentence:
# # #         sentence = "All fingers are visible."
# # #
# # #     # 添加 "and other fingers are visible."
# # #     sentence += ", and other fingers are visible."
# # #
# # #     return sentence
# joint_mapping = {
#     0: "Wrist",
#     1: "MCP of the thumb", 2: "PIP of the thumb", 3: "DIP of the thumb", 4: "TIP of the thumb",
#     5: "MCP of the index finger", 6: "PIP of the index finger", 7: "DIP of the index finger",
#     8: "TIP of the index finger",
#     9: "MCP of the middle finger", 10: "PIP of the middle finger", 11: "DIP of the middle finger",
#     12: "TIP of the middle finger",
#     13: "MCP of the ring finger", 14: "PIP of the ring finger", 15: "DIP of the ring finger",
#     16: "TIP of the ring finger",
#     17: "MCP of the pinky", 18: "PIP of the pinky", 19: "DIP of the pinky", 20: "TIP of the pinky"
# }
#
#
# class Text_deal():
#     def __init__(self, data_split):
#
#         self.data_split = data_split if data_split == 'train' else 'test'
#         self.root_dir = osp.join('/home/dataset/DEX_YCB')
#         self.annot_path = osp.join(self.root_dir, 'annotations')
#         self.json_path = '/home/zzh23/codes/gcn32z/data/DEX_YCB/'
#         self.clip, _ = clip.load("ViT-B/32", device='cuda', jit=False,
#                                  download_root='/home/zzh23/codes/gcn32z/CLIP-main')
#         self.long_clip, _ = longclip.load("/home/zzh23/codes/gcn32z/Long_CLIP_main/checkpoints/longclip-L.pt",
#                                           device='cuda')
#         # 加载 JSON 数据
#         self.load_json_data()
#
#     def load_json_data(self):
#         if self.data_split == 'train':
#             train_grasp_json_path = osp.join(self.json_path, "train_check_grasping.json")
#             train_object_json_path = osp.join(self.json_path, "train_grasp_object.json")
#             train_obj_occ_path = '/home/leon/HOAI/read_dexycb/train_object_occlusion_label.json'
#             train_self_occ_path = '/home/leon/HOAI/read_dexycb/train_self_occlusion_label.json'
#
#             with open(train_grasp_json_path, 'r') as file:
#                 self.train_if_grasp = json.load(file)
#             with open(train_object_json_path, 'r') as file:
#                 self.train_obj_name_dict = json.load(file)
#             with open(train_obj_occ_path, 'r') as file:
#                 self.train_object_obscured_data = json.load(file)
#             with open(train_self_occ_path, 'r') as file:
#                 self.train_self_obscured_data = json.load(file)
#         else:
#             test_grsap_json_path = osp.join(self.json_path, "test_check_grasping.json")
#             test_object_json_path = osp.join(self.json_path, "dexycb_test_grasp_object.json")
#             test_obj_occ_path = '/home/leon/HOAI/read_dexycb/test_object_occlusion_label.json'
#             test_self_occ_path = '/home/leon/HOAI/read_dexycb/test_self_occlusion_label.json'
#
#             with open(test_grsap_json_path, 'r') as file:
#                 self.test_if_grasp = json.load(file)
#             with open(test_object_json_path, 'r') as file:
#                 self.test_obj_name_dict = json.load(file)
#             with open(test_obj_occ_path, 'r') as file:
#                 self.test_object_obscured_data = json.load(file)
#             with open(test_self_occ_path, 'r') as file:
#                 self.test_self_obscured_data = json.load(file)
#
#
#     def load_data(self):
#         db = COCO(osp.join(self.annot_path, "DEX_YCB_s0_{}_data.json".format(self.data_split)))
#         data_dict = {}
#         for aid in tqdm(db.anns.keys(), desc='Loading data', total=len(db.anns)):
#             ann = db.anns[aid]
#             image_id = ann['image_id']
#             img = db.loadImgs(image_id)[0]
#             img_path = osp.join(self.root_dir, img['file_name'])
#             try:
#                 grasp = self.train_if_grasp[img_path] if self.data_split == 'train' else self.test_if_grasp[img_path]
#             except KeyError:
#                 grasp = -1
#             try:
#                 obj_name = self.train_obj_name_dict[img_path] if self.data_split == 'train' else \
#                 self.test_obj_name_dict[img_path]
#             except KeyError:
#                 obj_name = 'object'
#             obj_name = (re.sub(r'\d+', '', obj_name)).replace('_', ' ') if obj_name is not None else None
#
#             global_text = []
#             self_obscured_joints, object_obscured_joints, both_vis_joints, both_vis_joints_num = joints_deal(img_path,
#                                                                                                              self.train_object_obscured_data if self.data_split == 'train' else self.test_object_obscured_data,
#                                                                                                              self.train_self_obscured_data if self.data_split == 'train' else self.test_self_obscured_data)
#             if grasp == 1:
#                 global_text.append(f"{obj_name} is being held by the {ann['hand_type']} hand in the image.")
#             else:
#                 global_text.append(f"The {ann['hand_type']} hand is going to grasp the {obj_name} with curled fingers.")
#
#             focal_text = generate_focal_text(object_obscured_joints, self_obscured_joints, obj_name)
#
#             text_g_array = np.array(global_text).squeeze()
#             text_g_list = text_g_array.tolist()
#             text_f_array = np.array(focal_text).squeeze()
#             text_f_list = text_f_array.tolist()
#             with torch.no_grad():
#                 text_g_feature = clip.tokenize(text_g_list).cuda()
#                 text_f_feature = longclip.tokenize(text_f_list).cuda()
#                 text_g_features = self.clip.encode_text(text_g_feature.cuda())
#                 text_f_features = self.long_clip.encode_text(text_f_feature.cuda())
#
#             data_dict[img_path] = {
#                 "image_id": image_id,
#                 "both_vis_joints": both_vis_joints,
#                 "both_vis_joints_num": both_vis_joints_num,
#                 "global_text": global_text,
#                 'focal_text': focal_text,
#                 "text_g_features": text_g_features.cpu().numpy().tolist(),
#                 "text_f_features": text_f_features.cpu().numpy().tolist()
#             }
#         return data_dict
#
#
# def joints_deal(img_path, object_obscured_data, self_obscured_data):
#     if img_path in object_obscured_data:
#         object_obscured_joints = object_obscured_data[img_path]
#     else:
#         object_obscured_joints = {str(i): 0 for i in range(21)}
#     if img_path in self_obscured_data:
#         self_obscured_joints = self_obscured_data[img_path]
#     else:
#         self_obscured_joints = {str(i): 0 for i in range(21)}
#     both_vis_joints = [int(key) for key in self_obscured_joints if
#                        self_obscured_joints[key] == 0 and object_obscured_joints[key] == 0]
#     both_vis_joints_num = len(both_vis_joints)
#     return self_obscured_joints, object_obscured_joints, both_vis_joints, both_vis_joints_num
#
#
# def generate_sentence(object_obscured_joints, self_obscured_joints, obscured_by):
#     sentence = ""
#     both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints)
#     self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]
#     if object_obscured_joints:
#         if len(object_obscured_joints) == 1:
#             sentence += "The {} is obscured by {}, ".format(object_obscured_joints[0], obscured_by)
#         else:
#             obscured_joints_str = ", ".join(object_obscured_joints)
#             sentence += "{} are obscured by {}, ".format(obscured_joints_str, obscured_by)
#     if self_obscured_joints:
#         if len(self_obscured_joints) == 1:
#             sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
#         else:
#             self_obscured_joints_str = ", ".join(self_obscured_joints)
#             sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
#     sentence = sentence.rstrip(', ')
#     if not sentence:
#         sentence = "All fingers are visible."
#     sentence += ", and other fingers are visible."
#     return sentence
#
#
# def generate_focal_text(object_occlusion_json, self_occlusion_json, obj_name):
#     object_obscured_joints = [joint_mapping[int(i)] for i, obscured in object_occlusion_json.items() if
#                               str(obscured) != '0']
#     self_obscured_joints = [joint_mapping[int(i)] for i, obscured in self_occlusion_json.items() if
#                             str(obscured) == '1']
#     return generate_sentence(object_obscured_joints, self_obscured_joints, obj_name)
#
#
# # 创建 Text_deal 实例并加载数据
# Text_deal = Text_deal('train')
# data_dict = Text_deal.load_data()
#
# # 保存数据到 JSON 文件
# filename = 'train_ycb_text_data.json'
# with open(filename, 'w') as file:
#     json.dump(data_dict, file, indent=4)  # indent用于美化输出，使其更易于阅读
#
# print(f"数据已保存到 {filename}")
#
# # # def joints(object_obscured_joints,self_obscured_joints):
# #
# #     both_vis_joints = [int(key) for key in self_obscured_joints if self_obscured_joints[key] == 0 and object_obscured_joints[key] == 0]
# #     both_vis_joints_num = len(both_vis_joints)
# #     return self_obscured_joints,object_obscured_joints,both_vis_joints,both_vis_joints_num
# #
# #
#
# #
# # obj_name_mapping = {
# #     2 :'master chef can', 3 :'cracker box', 4:'sugar box', 5 :'tomato soup can', 6 : 'mustard bottle',
# #     7 : 'tuna fish can',8:'pudding box',9:'gelatin box',10:'potted meat can', 11:'banana', 19 :'pitcher base',
# #     21: 'bleach cleanser', 24: 'bowl', 25:'mug', 35 :'power drill',36 :'wood block', 37 :'scissors',
# #     40:'large marker',51: 'large clamp',52 :'extra large clamp',61: 'foam brick'
# # }
# # obj_text_mapping = {
# # 2 :'The can is a cylinder with mostly blue sides and a white lid',
# # 3 :'The box is a cuboid with mostly red',
# # 4:' The box is a yellow-and-white cuboid',
# # 5 :'The can is a red-and-white cylinder',
# # 6 : 'The bottle is mostly yellow',
# # 7 : 'The can is a cylinder with mostly blue sides and a silver lid',
# # 8:'The box is cuboid with mostly brown and white',
# # 9:'The box is cuboid with mostly red and white',
# # 10:'The can is mostly blue',
# # 11:'The banana is yellow',
# # 19 :'This kettle with a handle is mostly blue',
# # 21: 'The container is mostly white',
# # 24: 'The bowl is red',
# # 25: 'The mug with a handle is red',
# # 35 :'The drill is red-and-black',
# # 36 :'The block is cuboid with mostly yellow',
# # 37 :'The scissors is yellow and black',
# # 40:'The marker is white-and-black',
# # 51: 'The large clamp is black',
# # 52 :'The extra large clamp is black',
# # 61: 'The brick is a red cuboid'
# # }
# # object_occlusion_json = {
# #     "0": 0, "1": 1, "2": 1, "3": 1, "4": 0,
# #     "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
# #     "11": 0, "12": 0, "13": 0, "14": 0, "15": 0,
# #     "16": 0, "17": 0, "18": 0, "19": 0, "20": 0}
# # self_occlusion_json  = {"0": 1, "1": 0, "2": 1, "3": 1, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
# #                         "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 1, "20": 1}
# # obj_name = 'banana'
# # def generate_sentence(object_obscured_joints, self_obscured_joints, obscured_by_dict):
# #     # 初始化句子
# #     sentence = ""
# #     both_obscured_joints = set(object_obscured_joints) & set(self_obscured_joints)
# #
# #     self_obscured_joints = [joint for joint in self_obscured_joints if joint not in both_obscured_joints]
# #     # 假设 self_obscured_joints 是一个字典，其键是关节点，值是是否被自身遮挡（True 或 False）
# #     if object_obscured_joints:
# #
# #         # 存储不重复的遮挡信息
# #         unique_obscured_info = {}
# #         # 遍历字典，构建不重复的遮挡信息
# #         for point, object in obscured_by_dict.items():
# #             if object not in unique_obscured_info:
# #                 unique_obscured_info[object] = []
# #             unique_obscured_info[object].append(int(point))
# #
# #         # 构建最终的描述字符串
# #         descriptions = []
# #         for obj, points in unique_obscured_info.items():
# #             # 使用正确的字典名称 point_descriptions 来访问关键点描述
# #             point_descriptions_list = [joint_mapping[point] for point in points]
# #             # 将关键点描述列表转换为字符串
# #             if len(point_descriptions_list) == 1:
# #                 point_descriptions_str = ', '.join(point_descriptions_list)
# #                 sentence += "The {} is obscured by {}, ".format(point_descriptions_str,obj)
# #             else:
# #                 point_descriptions_str = ', '.join(point_descriptions_list)
# #                 sentence += "{} are obscured by {}, ".format(point_descriptions_str,obj)
# #                 # descriptions.append(f"{point_descriptions_str} are obscured by {obj}")
# #
# #         # 将所有描述合并为一个字符串
# #         # final_description = ' '.join(descriptions)
# #         # 处理自身遮挡的关节点
# #         # self_obscured_joints = ", ".join(self_obscured_joints)
# #     if self_obscured_joints:
# #         if len(self_obscured_joints) == 1:
# #             sentence += "and the {} is obscured by other fingers".format(self_obscured_joints[0])
# #         else:
# #             self_obscured_joints_str = ", ".join(self_obscured_joints)
# #             sentence += "and {} are obscured by other fingers".format(self_obscured_joints_str)
# #
# #     # 清除句子尾部多余的逗号或空格
# #     sentence = sentence.rstrip(', ')
# #
# #     # 如果句子为空，则表示没有遮挡
# #     if not sentence:
# #         sentence = "All fingers are visible."
# #
# #     # 添加 "and other fingers are visible."
# #     sentence += ", and other fingers are visible."
# #     return sentence
# # def generate_focal_text(object_occlusion_json, self_occlusion_json):
# #     # 提取被物体遮挡和被手部自身遮挡的关节点名称
# #     object_obscured_joints = [
# #         joint_mapping[int(i)] for i, obscured in object_occlusion_json.items() if str(obscured) != '0'
# #     ]
# #     self_obscured_joints = [
# #         joint_mapping[int(i)] for i, obscured in self_occlusion_json.items() if str(obscured) == '1'
# #     ]
# #
# #     mapped_objects = {}
# #     for key, value in object_occlusion_json.items():
# #         # 检查value是否在obj_name_mapping中
# #         if value in obj_name_mapping:
# #             mapped_objects[key] = obj_name_mapping[value]
# #
# #     # 生成描述句子
# #     focal_text = generate_sentence(object_obscured_joints, self_obscured_joints, mapped_objects)
# #
# #     return focal_text
# # # 调用函数生成文本
# # def generate_objects_text(object_occlusion_json):
# #     unique_descriptions = {}
# #     for key, obj_id in object_occlusion_json.items():
# #         # 检查obj_id是否在obj_text_mapping中有对应的描述
# #         if obj_id in obj_text_mapping:
# #             # 去掉描述前的"The "，如果存在的话
# #             description = obj_text_mapping[obj_id].replace("The ", "")
# #             # 如果描述是唯一的，添加到unique_descriptions字典中
# #             if description not in unique_descriptions.values():
# #                 unique_descriptions[key] = description
# #
# #     # 将unique_descriptions字典中的描述转换为列表
# #     descriptions = list(unique_descriptions.values())
# #
# #     # 构建描述列表，第一句使用"The"，其余使用"and the"
# #     descriptions_list = ["The " + desc for desc in descriptions]
# #     if len(descriptions) > 1:
# #         # 从第二句开始，使用"and the "作为前缀
# #         descriptions_list[1:] = ["and the " + desc for desc in descriptions[1:]]
# #
# #     # 将描述列表连接成一个字符串
# #     obj_text = ', '.join(descriptions_list)+'.'
# #
# #     return obj_text
# # obj_text = generate_objects_text(object_occlusion_json)
# # # focal_text = generate_focal_text(object_occlusion_json, self_occlusion_json)
# # print(obj_text)
# # # def joints_center(data,joints_img):
# # #     output_size = 32
# # #     joints_img_coords = []
# # #     # 遍历找到的键，累加坐标
# # #     for key in data['both_vis_joints']:
# # #         x, y = joints_img[key]
# # #         joints_img_coords.append([y, x])
# # #     joints_img_centers = []
# # #     joints_img_center = []
# # #
# # #     centers = np.sum(np.array(joints_img_coords), axis=0) / data['both_vis_joints_num'] * 32
# # #
# # #     x, y = int(centers[0]), int(centers[1])
# # #     new_x = x + choice([-1, 0, 1])
# # #     new_y = y + choice([-1, 0, 1])
# # #     if new_x < 0 or new_x >= output_size or new_y < 0 or new_y >= output_size:
# # #         new_x = x
# # #         new_y = y
# # #     x, y = new_x, new_y
# # #     joints_img_center.append([y, x])
# # #     joints_img_centers.append(np.array(joints_img_center))
# # #     joints_img_centers = np.concatenate(joints_img_centers, axis=0)
# # #     joints_img_centers = torch.from_numpy(joints_img_centers)
# # #     return joints_img_centers
# # """
#
#
# # focal_text = generate_focal_text(object_occlusion_json,self_occlusion_json,obj_name)
# # print(focal_text)
# # """




# 假设 joint_mapping 已经定义

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
mapping = {
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
class Text_YCB():
    '''
    Replace the "None" part with your own path
    '''
    def __init__(self, data_split):
        self.data_split = data_split if data_split == 'train' else 'test'
        self.root_dir = None ##### The root path of the dataset
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.json_path = None   ##### The root path of the json of the saved relevant information
        self.clip, _ = clip.load("ViT-B/32", device='cuda', jit=False,download_root=None) #####The path of the clip model
        self.long_clip, _ = longclip.load(None,device='cuda')  ####The path of the long-clip model
        self.load_json_data()

    def load_json_data(self):
        if self.data_split == 'train':
            train_grasp_object_occ_json_path = osp.join(self.json_path, "YCB_Train_OGG.json")

            with open(train_grasp_object_occ_json_path, 'r') as file:
                self.train_grasp_object_occ = json.load(file)
        else:
            test_grasp_object_occ_json_path = osp.join(self.json_path, "YCB_Test_OGG.json")

            with open(test_grasp_object_occ_json_path, 'r') as file:
                self.test_grasp_object_occ = json.load(file)

    def load_data(self):
        db = COCO(None) #Your own path
        data_dict = {}
        for aid in tqdm(db.anns.keys(), desc='Loading data', total=len(db.anns)):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            # img_path = osp.join(self.root_dir, img['file_name'])
            img_path = img['file_name']
            if self.data_split == 'train':
                self.grasp = self.train_grasp_object_occ[img_path]['grasp']
                self.obj_name = self.train_grasp_object_occ[img_path]['obj_name']
                self.train_object_occ_data = self.train_grasp_object_occ[img_path]['obj_occ']
                self.train_self_occ_data = self.train_grasp_object_occ[img_path]['self_occ']
            else:
                self.grasp = self.test_grasp_object_occ[img_path]['grasp']
                self.obj_name = self.test_grasp_object_occ[img_path]['obj_name']
                self.test_object_occ_data = self.test_grasp_object_occ[img_path]['obj_occ']
                self.test_self_occ_data = self.test_grasp_object_occ[img_path]['self_occ']

            global_text = []
            (self_obscured_joints, object_obscured_joints, both_vis_joints, both_vis_joints_num) \
                = joints_deal_YCB(self.train_object_occ_data if self.data_split == 'train' else self.test_object_occ_data,
                                       self.train_self_occ_data if self.data_split == 'train' else self.test_self_occ_data)
            if self.grasp == 1:
                global_text.append(f"The{self.obj_name} is being held by the {ann['hand_type']} hand in the image.")
            else:
                global_text.append(f"The {ann['hand_type']} hand is going to grasp the {self.obj_name} with curled fingers.")

            local_text = generate_local_text_YCB(object_obscured_joints, self_obscured_joints, self.obj_name)

            text_g_array = np.array(global_text).squeeze()
            text_g_list = text_g_array.tolist()
            text_l_array = np.array(local_text).squeeze()
            text_l_list = text_l_array.tolist()
            with torch.no_grad():
                text_g_feature = clip.tokenize(text_g_list).cuda()
                text_l_feature = longclip.tokenize(text_l_list).cuda()
                text_g_features = self.clip.encode_text(text_g_feature.cuda())
                text_l_features = self.long_clip.encode_text(text_l_feature.cuda())
            data_dict[img_path] = {
                "image_id": image_id,
                "both_vis_joints": both_vis_joints,
                "both_vis_joints_num": both_vis_joints_num,
                "global_text": global_text,
                'focal_text': local_text,
                "text_g_features": text_g_features.cpu().numpy().tolist(),
                "text_l_features": text_l_features.cpu().numpy().tolist(),
            }
        return data_dict

class Text_HO3D():
    def __init__(self, train):
        self.root_dir = None ##### The root path of the dataset
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.clip, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root=None)##### The root path of the json of the saved relevant information
        self.long_clip, _ = longclip.load(None,device='cuda')####The path of the long-clip model
        self.json_path = None ##### The root path of the json of the saved relevant information

        self.load_json()
    def load_json(self):
        grasp_object_occ_json_path = osp.join(self.json_path, "HO3D_OGG.json")
        with open(grasp_object_occ_json_path, 'r') as file:
            self.grasp_object_occ = json.load(file)
    def load_data(self):
        db = COCO(None) #Your own path
        data_dict = {}
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, 'train', img['file_name'])

            file_name = img['file_name']
            folder_name = file_name.split('/')[0]
            self.obj_name = mapping.get(folder_name, "Unknown")
            self.grasp = self.grasp_object_occ[img_path]['grasp']
            self.object_occ_data = self.grasp_object_occ[img_path]['obj_occ']
            self.self_occ_data = self.grasp_object_occ[img_path]['self_occ']

            train_self_obscured_joints, train_object_obscured_joints, both_vis_joints, both_vis_joints_num = joints_deal_HO3D(
                self.object_occ_data, self.self_occ_data)
            global_text = []
            if self.grasp == 1:

                global_text.append('The {} is being held by the right hand in the image.'.format(self.obj_name))
            else:
                global_text.append(
                    'The right hand is going to grasp the {} with curled fingers in the image.'.format(self.obj_name))

            local_text = generate_local_text_HO3D(train_object_obscured_joints, train_self_obscured_joints,self.obj_name)
            text_g_array = np.array(global_text).squeeze()
            text_g_list = text_g_array.tolist()
            text_l_array = np.array(local_text).squeeze()
            text_l_list = text_l_array.tolist()
            with torch.no_grad():
                text_g_feature = clip.tokenize(text_g_list).cuda()
                text_l_feature = longclip.tokenize(text_l_list).cuda()
                text_g_features = self.clip.encode_text(text_g_feature.cuda())
                text_l_features = self.long_clip.encode_text(text_l_feature.cuda())

            data_dict[img_path] = {
                "image_id": image_id,
                "both_vis_joints": both_vis_joints,
                "both_vis_joints_num": both_vis_joints_num,
                "global_text": global_text,
                'local_text': local_text,
                "text_g_features": text_g_features.cpu().numpy().tolist(),
                "text_l_features": text_l_features.cpu().numpy().tolist(),
            }
        return data_dict

# Create an instance of YCB_Text and load the data
# '''
Text_deal_ycb = Text_YCB('train')
data_dict = Text_deal_ycb.load_data()
filename = 'demo.json'  ### Your own output path
with open(filename, 'w') as file:
    json.dump(data_dict, file, indent=0) 
#
# print(f"Data have saved in {filename}")
# '''

# Create an instance of HO3D_Text and load the data
'''
HO3D_text_deal = Text_HO3D('test')
data_dict = HO3D_text_deal.load_data()
filename = None  ### Your own output path
with open(filename, 'w') as file:
    json.dump(data_dict, file, indent=0)  #
print(f"Data have saved in {filename}")
# '''


