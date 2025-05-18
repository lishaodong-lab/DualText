import numpy as np
import math


class HeatmapGenerator():
    def __init__(self, output_res):
        self.output_res = output_res
        # self.num_keypoints = 21

    def gaussian2D(self, shape, sigma=1.):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    # def is_vis(both_vis_joints):
    #     visibility_mask = [0] * 21
    #     for joint_index in both_vis_joints:
    #         visibility_mask[joint_index] = 1
    #     return visibility_mask

    def __call__(self, keypoints, bbox,both_vis_joints):

        num_keypoints = len(keypoints)

        hms = np.zeros((num_keypoints, self.output_res, self.output_res), dtype=np.float32)
        mask = np.ones((num_keypoints, 1, 1), dtype=np.float32)


        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height


        bbox_center_x = xmin + width / 2
        bbox_center_y = ymin + height / 2

        # 根据边界框的尺寸计算高斯热图的 radius
        radius = gaussian_radius((math.ceil(height), math.ceil(width)))
        radius = max(0, int(radius))
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        visibility_mask = is_vis(both_vis_joints)
        for idx, (x, y) in enumerate(keypoints):


            if visibility_mask[idx]>0:
                # x, y = int(keypoints[0]), int(keypoints[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:continue

                left, right = min(int(x), radius), min(self.output_res - int(x), radius + 1)
                top, bottom = min(int(y), radius), min(self.output_res - int(y), radius + 1)

                masked_heatmap = hms[idx][int(y) - top:int(y) + bottom, int(x) - left:int(x) + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
                    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
            else:
                mask[idx] = 0.0
        return hms, mask

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    discriminant = b3 ** 2 - 4 * a3 * c3
    sq3 = np.sqrt(max(discriminant, 0))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def is_vis(both_vis_joints):
    visibility_mask = [0] * 21
    for joint_index in both_vis_joints:
        visibility_mask[joint_index] = 1
    return visibility_mask

def bbox_to_corners(bbox):

    xmin, ymin, width, height = bbox

    corners = np.array([[xmin, ymin],
                        [xmin + width, ymin],
                        [xmin + width, ymin + height],
                        [xmin, ymin + height]])
    corners_reshaped = corners.reshape(1, 4, 2)
    return corners_reshaped



