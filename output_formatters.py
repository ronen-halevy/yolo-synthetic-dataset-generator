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
from PIL import Image, ImageDraw
import math
import yaml
import json
from datetime import date, datetime
import random
import argparse
import os
from pathlib import Path

# create a row labels text file per image. format:
# x0l,y0l,x0h,y0h,c, ......
# .
# xnl,ynl,xnh,ynh,c

def create_per_image_labels_files(images_paths, images_bboxes, images_sizes, images_objects_categories_indices,

                                  output_dir):
    """

    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_sizes:
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    print('create_per_image_labels_files')
    output_dir = f'{output_dir}labels/'
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # directory already exists
        pass
    for bboxes, image_path, images_size, categories_indices in zip(images_bboxes, images_paths, images_sizes,
                                                              images_objects_categories_indices):
        im_height = images_size[0]
        im_width = images_size[1]

        head, filename = os.path.split(image_path)
        labels_filename = f"{output_dir}{filename.rsplit('.', 1)[0]}.txt"
        with open(labels_filename, 'w') as f:
            for bbox, category_id in zip(bboxes, categories_indices):
                bbox_arr = np.array(bbox, dtype=float)
                xcycwh_bbox = [(bbox_arr[0] + bbox_arr[2] / 2) / im_width, (bbox_arr[1] + bbox_arr[3] / 2) / im_height,
                               bbox_arr[2] / im_width, bbox_arr[3] / im_height]
                entry = f"{category_id} {' '.join(str(e) for e in xcycwh_bbox)}\n"
                f.write(entry)


# create a row labels text file. format:
# imag1_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c
# .
# imagm_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c

def create_row_text_labels_file(images_paths, images_bboxes, images_objects_categories_indices,
                                output_dir):
    """
    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    print('create_row_text_labels_file')

    entries = []
    for image_path, categories_indices, bboxes in zip(images_paths, images_objects_categories_indices, images_bboxes):

        # image_path = f'{output_dir}/{image_path} '
        entry = f'{image_path} '
        for bbox, category_id in zip(bboxes, categories_indices):
            bbox_arr = np.array(bbox, dtype=float)
            xyxy_bbox = [bbox_arr[0], bbox_arr[1], bbox_arr[0] + bbox_arr[2], bbox_arr[1] + bbox_arr[3]]
            for vertex in xyxy_bbox:
                entry = f'{entry}{vertex},'
            entry = f'{entry}{float(category_id)} '
        entries.append(entry)
        opath = f'{output_dir}/all_entries.txt'
        file = open(opath, 'w')
        for item in entries:
            file.write(item + "\n")
        file.close()



# Create a coco like format label file. format:
# a single json file with 4 tables:
#     "info":
#     "licenses":
#     "images": images_records,
#     "categories": categories_records,
#     "annotations": annotatons_records
def create_coco_labels_file(images_paths, images_sizes, images_bboxes, images_objects_categories_indices,
                            category_names, super_category_names, annotations_output_path):
    """
     :param images_paths: list of dataset image filenames
    :param images_sizes: list of per image [im.height, im.width] tuples
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices: list of per image categories_indices arrays
    :param category_names: list of all dataset's category names
    :param super_category_names:  list of all dataset's super_category_names
    :param annotations_output_path: path for output file storage
    :return:
    """

    print('create_coco_labels_file')

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    # map_categories_id = {}

    # fill category
    id = 0
    for category_name, supercategory in zip(category_names, super_category_names):

        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": supercategory,
                "id": id,
                "name": category_name,
            })
            added_category_names.append(category_name)
            # map_categories_id.update({category_name: id})
            id += 1

    images_records = []
    annotatons_records = []
    for example_id, (image_path, image_size, bboxes, objects_categories_indices) in enumerate(
            zip(images_paths, images_sizes, images_bboxes, images_objects_categories_indices)):

        # images records:

        images_records.append({
            "license": '',
            "file_name": image_path,
            "coco_url": "",
            'width': image_size[1],
            'height': image_size[0],
            "date_captured": str(datetime.now()),
            "flickr_url": "",
            "id": example_id
        })

        # annotatons_records
        for bbox, category_id in zip(bboxes, objects_categories_indices):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": category_id,
                "id": anno_id
            }
            )
            anno_id += 1
    date_today = date.today()
    info = {
        "description": " Dataset",
        "url": '',
        # "version": config.get('version', 1.0),
        "year": date_today.year,
        "contributor": '',
        "date_created": str(date_today),
        "licenses": '',
        "categories": categories_records
    }
    output_records = {
        "info": info,
        "licenses": [],
        "images": images_records,
        "categories": categories_records,
        "annotations": annotatons_records
    }
    print(f'Save annotation  in {annotations_output_path}')
    with open(annotations_output_path, 'w') as fp:
        json.dump(output_records, fp)


