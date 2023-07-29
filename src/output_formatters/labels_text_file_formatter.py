#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_shapes_dataset.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#
# ================================================================

import numpy as np


# create a row labels text file. format:
# imag1_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c
# imagm_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c

def create_row_text_labels_file(images_paths, images_bboxes, images_objects_categories_indices,
                                labels_file_path):
    """
    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param labels_file_path: path of output labels text files
    :return:
    """
    print('create_row_text_labels_file')

    entries = []
    for image_path, categories_indices, bboxes in zip(images_paths, images_objects_categories_indices, images_bboxes):

        entry = f'{image_path} '
        for bbox, category_id in zip(bboxes, categories_indices):
            bbox_arr = np.array(bbox, dtype=float)
            xyxy_bbox = [bbox_arr[0], bbox_arr[1], bbox_arr[0] + bbox_arr[2], bbox_arr[1] + bbox_arr[3]]
            for vertex in xyxy_bbox:
                entry = f'{entry}{vertex},'
            entry = f'{entry}{float(category_id)} '
        entries.append(entry)
        file = open(labels_file_path, 'w')
        for item in entries:
            file.write(item + "\n")
        file.close()



