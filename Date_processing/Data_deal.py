import argparse
import numpy as np
import os
import csv
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class statistics:

    def __init__(self, csv_path, data_anno_rules):
        self.data_anno_rules = data_anno_rules
        self.csv_path = csv_path
        self.category_num_dict = {}

    def statistics_category_num(self):
        labelfiles = os.listdir(self.csv_path)
        for labelfile in labelfiles:
            label_path = self.csv_path + labelfile
            with open(label_path, 'r') as f:
                reader = csv.reader(f)
                for i in reader:
                    class_ = list(self.data_anno_rules .keys())[list(self.data_anno_rules .values()).index(int(float(i[1])))]
                    if class_ in self.category_num_dict.keys():
                        self.category_num_dict[class_] = self.category_num_dict[class_] + 1
                    else:
                        self.category_num_dict[class_] = 1
        print(self.category_num_dict)

    def statistics_distance(self):
        labelfiles = os.listdir(self.csv_path)
        len = ['0','10','20','30','40','50','60','70','80','90']
        len_80_90, len_70_80, len_60_70, len_50_60 =  [], [], [], []
        len_40_50 = []
        len_30_40 = []
        len__80__90 = []
        len__70__80 = []
        len_30 = []
        for file in tqdm(labelfiles):
            with open(self.csv_path + '/' + file, 'r') as f:
                reader = csv.reader(f)
                for i in reader:
                    if float(i[2]) > 8000:
                        len_80_90.append(float(i[2]))
                    if float(i[2]) < 8000 and float(i[2]) > 7000:
                        len_70_80.append(float(i[2]))
                    if float(i[2]) < 7000 and float(i[2]) > 6000:
                        len_60_70.append(float(i[2]))
                    if float(i[2]) < 6000 and float(i[2]) > 5000:
                        len_50_60.append(float(i[2]))
                    if float(i[2]) < 5000 and float(i[2]) > 4000:
                        len_40_50.append(float(i[2]))
                    if float(i[2]) < 4000 and float(i[2]) > 3000:
                        len_30_40.append(float(i[2]))
                    if float(i[2]) < 3000 :
                        len_30.append(float(i[2]))
                    if float(i[2]) > -9000 and float(i[2]) < -8000:
                        len__80__90.append(float(i[2]))
                    if float(i[2]) > -8000 and float(i[2]) < -7000:
                        len__70__80.append(float(i[2]))

        print(len(len_80_90))
        print(len(len_70_80))
        print(len(len_60_70))
        print(len(len_50_60))
        print(len(len_40_50))
        print(len(len_30_40))
        print(len(len__80__90))
        print(len(len__70__80))


class draw_img:
    def __init__(self):
        self.points_to_draw = []
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.z_min = 0
        self.z_max = 0

    def new_img(self, img_with, img_height, img_color):
        img = Image.new('RGB', (img_with, img_height), img_color)
        return img

    def light_point(self, points_to_draw, img_side, img_down, img_forward):
        tempx = (points_to_draw[:, 0] - self.x_min) * 30 + 30
        tempy = (points_to_draw[:, 1] - self.y_min) * 30 + 30
        tempz = (abs(points_to_draw[:, 2]) - abs(self.z_max)) * 30 + 30

        for ii in range(len(tempx)):
            img_side.putpixel((int(tempx[ii]), int(tempz[ii])), (255, 255, 0))
        for ii in range(len(tempx)):
            img_down.putpixel((int(tempx[ii]), int(tempy[ii])), (255, 255, 0))
        for ii in range(len(tempx)):
            img_forward.putpixel((int(tempy[ii]), int(tempz[ii])), (255, 255, 0))


    def add_words(self, add_img, words, word_size, word_x, word_y):
        fontText = ImageFont.truetype("simsun.ttc", encoding="utf-8", size=word_size)
        draw_all_word = ImageDraw.Draw(add_img)
        draw_all_word.text((word_x, word_y), words, font=fontText)

    def cal_img_size(self, points_to_draw):
        self.x_min = min(np.array(points_to_draw)[:, 0])
        self.x_max = max(np.array(points_to_draw)[:, 0])
        self.y_min = min(np.array(points_to_draw)[:, 1])
        self.y_max = max(np.array(points_to_draw)[:, 1])
        self.z_min = min(np.array(points_to_draw)[:, 2])
        self.z_max = max(np.array(points_to_draw)[:, 2])

        img_side_x = abs(int(self.x_max - self.x_min)) * 30 + 60
        img_side_y = abs(int(self.z_max - self.z_min)) * 30 + 120
        img_down_x = abs(int(self.x_max - self.x_min)) * 30 + 60
        img_down_y = abs(int(self.y_max - self.y_min)) * 30 + 120
        img_forward_x = abs(int(self.y_max - self.y_min)) * 30 + 120
        img_forward_y = abs(int(self.z_max - self.z_min)) * 30 + 120

        if img_side_x < 600:
            img_side_x = 600
        else:
            img_side_x = img_side_x

        if img_side_y < 300:
            img_side_y = 300
        else:
            img_side_y = img_side_y

        return img_side_x, img_side_y, img_down_x, img_down_y, img_forward_x, img_forward_y

    def slt_points_to_draw(self, gt_boxes, points, point_indices):
        num_obj = gt_boxes.shape[0]
        for i in range(num_obj):
            gt_points = points[point_indices[:, i]]
            if gt_points is not None:
                self.points_to_draw.append(gt_points)
        return self.points_to_draw


class select_box_point:

    def __init__(self, labelfile, lidar_path, csv_path):
        self.labelfile = labelfile
        self.lidar_path = lidar_path
        self.csv_path = csv_path
        self.infos = {}
        self.gt_boxes = []
        self.gt_boxes_all = []
        self.points = []
        self.PI_rads = math.pi / 180


    def infos_points(self):
        with open(self.lidar_path, 'r') as f:
            for line in f.readlines()[11:len(f.readlines()) - 1]:
                linestr = line.split(" ")
                if len(linestr) in [3, 4]:
                    linestr_convert = list(map(float, linestr))
                    self.points.append(linestr_convert)
        self.infos['points'] = np.array(self.points)

    def infos_gt_boxes(self):
        label_fpath = os.path.join(self.csv_path, self.labelfile)
        with open(label_fpath, 'r') as f:
            reader = csv.reader(f)
            for i in reader:
                if int(i[1]) == 1:
                    continue
                self.gt_boxes.append(
                    [float(i[2]) / 100, float(i[3]) / 100, float(i[4]) / 100, float(i[7]) / 100, float(i[8]) / 100,
                     float(i[9]) / 100, float(i[6]) * self.PI_rads])  # 依次为x,y,z,l,w,h,angle，单位为厘米和角度，请对照自己的标签格式自行修改
                self.gt_boxes_all.append(i)
        gt_boxes = np.array(self.gt_boxes)
        gt_boxes_all = np.array(self.gt_boxes_all)
        self.infos['gt_boxes'] = gt_boxes
        self.infos['gt_boxes_all'] = gt_boxes_all



class points_in_rbbox:

    def __init__(self, points, gt_boxes):
        self.axis = 2
        self.origin = (0.5, 0.5, 0)
        self.points = points
        self.gt_boxes = gt_boxes

    def points_in_rbbox(self):
        rbbox_corners = self.center_to_corner_box3d()  ##由中心点左边及长宽高计算出长方体角点
        surfaces = self.corner_to_surfaces_3d(rbbox_corners)
        indices = self.points_in_convex_polygon_3d_jit(surfaces)  ##计算长方体面内点的个数
        return indices

    def center_to_corner_box3d(self):
        corners = self.corners_nd()  # corners: [N, 8, 3]
        angles = self.gt_boxes[:, 6]
        centers = self.gt_boxes[:, :3]
        if angles is not None:
            corners = self.rotation_3d_in_axis(corners, angles)
        corners += centers.reshape([-1, 1, 3])

        for i in range(len(corners)):
            z_h = max(corners[i, :, -1])
            z_0 = min(corners[i, :, -1])
            z_max = z_h - (z_h - z_0) / 2
            z_min = z_0 - (z_h - z_0) / 2
            corners[i, [0, 3, 4, 7], -1] = z_min
            corners[i, [1, 2, 5, 6], -1] = z_max
        return corners

    def corners_nd(self):
        dims = self.gt_boxes[:, 3:6]
        origin = self.origin
        ndim = int(dims.shape[1])
        corners_norm = np.stack(
            np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
            axis=1).astype(dims.dtype)
        if ndim == 2:
            # generate clockwise box corners
            corners_norm = corners_norm[[0, 1, 3, 2]]
        elif ndim == 3:
            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
        corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
            [1, 2 ** ndim, ndim])
        return corners

    def rotation_3d_in_axis(self, points, angles):
        rot_sin = np.sin(angles)
        rot_cos = np.cos(angles)
        ones = np.ones_like(rot_cos)
        zeros = np.zeros_like(rot_cos)
        if self.axis == 1:
            rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                                  [rot_sin, zeros, rot_cos]])
        elif self.axis == 2 or self.axis == -1:
            rot_mat_T = np.stack([[rot_cos, rot_sin, zeros],
                                  [-rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
        elif self.axis == 0:
            rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                                  [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
        else:
            raise ValueError('axis should in range')
        new_points = np.einsum('aij,jka->aik', points, rot_mat_T)
        return new_points

    def corner_to_surfaces_3d(self, corners):
        surfaces1 = np.array([
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ])
        surfaces = surfaces1.transpose([2, 0, 1, 3])
        return surfaces

    def points_in_convex_polygon_3d_jit(self, polygon_surfaces, num_surfaces=None):
        num_polygons = polygon_surfaces.shape[0]  ##目标框个数
        if num_surfaces is None:
            num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
        normal_vec, d = self.surface_equ_3d(polygon_surfaces[:, :, :3, :])  ##求面的法向量
        return self._points_in_convex_polygon_3d_jit(polygon_surfaces, normal_vec, d, num_surfaces)  ##根据长方体内部点与面的法向量之间的关系求得在目标框内部的点

    def _points_in_convex_polygon_3d_jit(self, polygon_surfaces, normal_vec, d, num_surfaces):

        max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
        num_points = self.points.shape[0]
        num_polygons = polygon_surfaces.shape[0]
        ret = np.ones((num_points, num_polygons), dtype=np.bool_)

        for i in range(num_points):
            for j in range(num_polygons):
                for k in range(max_num_surfaces):
                    if k > num_surfaces[j]:
                        break
                    sign = (
                            self.points[i, 0] * normal_vec[j, k, 0] +
                            self.points[i, 1] * normal_vec[j, k, 1] +
                            self.points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                    if sign >= 0:
                        ret[i, j] = False
                        break
        return ret

    def surface_equ_3d(self, polygon_surfaces):
        surface_vec = polygon_surfaces[:, :, :2, :] - \
                      polygon_surfaces[:, :, 1:3, :]
        normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
        d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
        return normal_vec, -d



class save_data:
    def __init__(self, csv_path, points_path, yuanduan_boxes, yuanduan_points):
        self.csv_path = csv_path + 'save_gtboxes.csv'
        self.points_path = points_path
        self.yuanduan_boxes = yuanduan_boxes
        self.yuanduan_points = yuanduan_points


    def yuanduan_boxes_to_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.yuanduan_boxes)

    def yuanduan_points_to_csv(self):
        for i in range(len(self.yuanduan_points)):
            yuanduan_points = self.yuanduan_points[i]
            point_path = self.points_path + str(i) +'.csv'
            with open(point_path, 'a+') as f:
                writer = csv.writer(f)
                writer.writerows(yuanduan_points)
