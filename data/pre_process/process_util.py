import numpy as np
import cv2
import random
import torch
import math
import torchvision
import yaml
from pathlib import Path
import os
def R_T_to_matrix(R, T):
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = R
    transform_matrix[0:3, 3] = T
    return transform_matrix

def rvec_tvec_to_matrix(rvec, tvec):
    transform_matrix = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape((3,))
    transform_matrix[0:3, 0:3] = R
    transform_matrix[0:3, 3] = T
    return transform_matrix

def matrix_to_rvec_tvec(transform_matrix):
    R = transform_matrix[0:3, 0:3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = transform_matrix[0:3, 3]
    return rvec, tvec

def matrix_to_R_T(transform_matrix):
    R = transform_matrix[0:3, 0:3]
    T = transform_matrix[0:3, 3]
    return R, T

def rotate_translate_point_cloud(point_cloud, trans_matrix):
    homogeneous_coordinates = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    new_coordinates = np.dot(trans_matrix, homogeneous_coordinates.T).T[:,:3]
    return new_coordinates

def are_signs_opposite(points):
    points_max_signs = np.sign(points.max(axis=0))
    points_min_signs = np.sign(points.min(axis=0))
    if points_max_signs[0] * points_min_signs[0] < 0 or points_max_signs[1] * points_min_signs[1] < 0 or \
            points_max_signs[2] * points_min_signs[2] < 0:
        return True
    else:
        return False
def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = (xmax - xmin) * expansion_factor;
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width

    y_center = (ymin + ymax) / 2.;
    height = (ymax - ymin) * expansion_factor;
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)

    return bbox

def get_pkl_path(image_path):

    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    image_dir = os.path.dirname(os.path.abspath(image_path))


    meta_dir = os.path.normpath(os.path.join(image_dir, '..', 'meta'))


    pkl_filename = f'{image_filename}.pkl'
    pkl_path = os.path.join(meta_dir, pkl_filename)

    return pkl_path
def process_bbox(bbox, img_width, img_height, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    # aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * expansion_factor
    bbox[3] = h * expansion_factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox

def compute_bounding_box_corners(point_cloud):
    """Calculate the axial alignment bounding box of the point cloud"""
    return np.array([
        [point_cloud[:, 0].min(), point_cloud[:, 1].min(), point_cloud[:, 2].min()],
        [point_cloud[:, 0].max(), point_cloud[:, 1].max(), point_cloud[:, 2].max()]
    ])

def is_point_inside_cube(point, bbox):
    """Check whether the point is within the axis alignment bounding box"""
    return (bbox[0, 0] <= point[0] <= bbox[1, 0]) and \
        (bbox[0, 1] <= point[1] <= bbox[1, 1]) and \
        (bbox[0, 2] <= point[2] <= bbox[1, 2])

def load_yaml_config(file_path):
    """load YAML files"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
