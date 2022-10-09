import numpy as np
import os
from tqdm import tqdm
from Data_deal import select_box_point, points_in_rbbox, draw_img
from anno_document import *


def draw_gt(root_path_csv, root_path_pcd,  save_root_path, anno_class):
    labelfiles = os.listdir(root_path_csv)
    img_title_height = 60
    tri_title_height = 30
    for file in tqdm(labelfiles):
        pcd_file = root_path_pcd + file.split('.')[0] + '.pcd'
        slt = select_box_point(labelfile=file, lidar_path=pcd_file, csv_path=root_path_csv)
        slt.infos_points()
        slt.infos_gt_boxes()

        infos = slt.infos
        points = infos['points']
        gt_boxes = infos['gt_boxes']
        gt_boxes_all = infos['gt_boxes_all']

        pir = points_in_rbbox(points, gt_boxes)
        point_indices = pir.points_in_rbbox()

        draw_images = draw_img()
        draw_images.slt_points_to_draw(gt_boxes, points, point_indices)
        points_to_draw = draw_images.points_to_draw

        for i in range(len(points_to_draw)):
            if len(np.array(points_to_draw[i])[:, 0]) == 0:
                print('file_name: ', file, ', num: ', i)
                continue

            img_side_x, img_side_y, img_down_x, img_down_y, img_forward_x, img_forward_y = draw_images.cal_img_size(points_to_draw[i])
            img_side = draw_images.new_img(img_with=img_side_x, img_height=img_side_y, img_color='black')
            img_down = draw_images.new_img(img_with=img_side_x, img_height=img_down_y, img_color='black')
            img_forward = draw_images.new_img(img_with=img_forward_x, img_height=img_forward_y, img_color='black')
            draw_images.light_point(points_to_draw=points_to_draw[i], img_side=img_side, img_down=img_down, img_forward=img_forward)

            img_title = draw_images.new_img(img_with = img_side_x + img_side_x, img_height = img_title_height, img_color = 'black')
            draw_images.add_words(add_img=img_title, words=file, word_size=20, word_x=0, word_y=10)
            img_target_information = draw_images.new_img(img_with = img_side_x, img_height = tri_title_height, img_color = 'black')
            draw_images.add_words(add_img=img_target_information, words='目标信息', word_size=20, word_x=0, word_y=0)
            img_side_word = draw_images.new_img(img_with=img_side_x, img_height=tri_title_height, img_color='black')
            draw_images.add_words(add_img=img_side_word, words='侧视图', word_size=20, word_x=int(img_side_x / 2 - 50),word_y=10)
            img_down_word = draw_images.new_img(img_with=img_side_x, img_height=tri_title_height, img_color='black')
            draw_images.add_words(add_img=img_down_word, words='俯视图', word_size=20, word_x=int(img_side_x / 2 - 50),word_y=10)
            img_forward_word = draw_images.new_img(img_with=img_side_x, img_height=tri_title_height, img_color='black')
            draw_images.add_words(add_img=img_forward_word, words='正视图', word_size=20, word_x=int(img_side_x / 2 - 50), word_y=10)
            img_all_word = draw_images.new_img(img_with=img_side_x, img_height=img_side_y, img_color='black')
            draw_images.add_words(add_img=img_all_word,
                                  words='\n' + '目标类别: ' + list(anno_class.keys())[list(anno_class.values()).index(int(gt_boxes_all[i][1]))] + '\n\n' + '目标长度: 长: ' + str(
                                      round(gt_boxes[i][3], 2)) + ' 宽: ' + str(round(gt_boxes[i][4], 2)) + ' 高: ' + str(
                                      round(gt_boxes[i][5], 2)) + '\n\n' +
                                        '目标位置: X: ' + str(round(gt_boxes[i][0], 2)) + ' Y: ' + str(
                                      round(gt_boxes[i][1], 2)) + ' Z: ' + str(round(gt_boxes[i][2], 2)),
                                  word_size=20, word_x=0, word_y=0)

            img_all = draw_images.new_img(img_with=img_side_x + img_side_x, img_height=img_side_y + img_down_y + img_forward_y,  img_color='black')
            img_all.paste(img_title, (0, 0))
            img_all.paste(img_target_information, (0, img_title_height))
            img_all.paste(img_all_word, (0, img_title_height + tri_title_height))
            img_all.paste(img_side_word, (img_side_x, img_title_height))
            img_all.paste(img_side, (img_side_x, img_title_height + tri_title_height))

            img_all.paste(img_forward_word, (0, img_title_height + tri_title_height + img_side_y))
            img_all.paste(img_forward, (int(img_side_x/2 - img_forward_x/2), img_title_height + tri_title_height + img_side_y + tri_title_height))
            img_all.paste(img_down_word, (img_side_x, img_title_height + tri_title_height + img_side_y))
            img_all.paste(img_down, (img_side_x, img_title_height + tri_title_height + img_side_y + tri_title_height))

            save_png_path = save_root_path + file.split('.')[0] + '__' + str(i) + '.png'
            img_all.save(save_png_path)


if __name__ == '__main__':
    root_path_pcd = '/media/wanji/lijingyang/docker/ezhou_data/test/pcd/'
    root_path_csv = '/media/wanji/lijingyang/docker/ezhou_data/test/csv/'
    save_root_path = '/media/wanji/lijingyang/docker/ezhou_data/test/gt_database/'
    if not os.path.isdir(save_root_path):
        os.makedirs(save_root_path)
    draw_gt(root_path_csv, root_path_pcd, save_root_path, file_20211009)