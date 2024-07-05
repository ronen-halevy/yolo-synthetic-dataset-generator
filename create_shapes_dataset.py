#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : main_create_shapes_dataset.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#   main method for  shapes dataset creation.
#   1. reads splits sizes and destination output path from config.json
#   2. Creates an instance of  ShapesDataset and generates the dataset
#   3. Envokes formatters to save dataset labels in various formats e.g. coco, multi text file (yolov5 ultralics like),
#   single text file
# ================================================================
import os

import yaml
import argparse
from pathlib import Path
import numpy as np

from src.create_label_files import (create_coco_json_lable_files,
                                    create_detection_entries, entries_to_files, dota_entries_to_files,create_segmentation_label_files, create_detection_kpts_entries,
                                    create_detection_labels_unified_file)
from src.shapes_dataset import ShapesDataset

def xywh2xyxy(obboxes, theta):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    cls,  center, w, h = np.split(obboxes, (1, 3, 4), axis=-1)




    point1 = center + np.concatenate([-w/2,-h/2], axis=1)
    point2 = center + np.concatenate([w/2,-h/2], axis=1)
    point3 = center + np.concatenate([w/2,h/2], axis=1)
    point4 = center + np.concatenate([-w/2,h/2], axis=1)

    # order = obboxes.shape[:-1]
    return np.concatenate(
            [point1, point2, point3, point4, cls], axis=-1)


import math

def rotate(hbboxes, theta0):
    rot_angle = theta0 / 180 * math.pi  # rot_tick*np.random.randint(0, 8)

    rotate_bbox = lambda xy: np.concatenate([np.sum(xy * (math.cos(rot_angle),  math.sin(rot_angle)), axis=-1,keepdims=True),
                              np.sum(xy * (-math.sin(rot_angle) , math.cos(rot_angle)), axis=-1,keepdims=True)], axis=-1)



    offset_xy = (np.max(hbboxes, axis=-2) + np.min(hbboxes,axis=-2)) / 2
    hbboxes_ = hbboxes - offset_xy
    rbboxes =  rotate_bbox(hbboxes_)
    rbboxes=rbboxes+offset_xy
    return rbboxes

# def rotateo(hbboxes, theta0):
#     rot_angle = theta0 / 180 * math.pi  # rot_tick*np.random.randint(0, 8)
#     # rot_angle = 0  # !!!todo debug ronen!!!!!!
#     # oo=sin(rot_angle)
#     rotate_x = lambda x, y: x * math.cos(rot_angle) + y * math.sin(rot_angle)
#     rotate_y = lambda x, y: -x * math.sin(rot_angle) + y * math.cos(rot_angle)
#     x_offset = (np.max(hbboxes[:, [0, 2, 4, 6]]) + np.min(hbboxes[:, [0, 2, 4, 6]])) / 2
#     y_offset = (np.max(hbboxes[:, [1, 3, 5, 7]]) + np.min(hbboxes[:, [1, 3, 5, 7]])) / 2
#     x_ = hbboxes[:, [0, 2, 4, 6]] - x_offset
#     y_ = hbboxes[:, [1, 3, 5, 7]] - y_offset
#     x_, y_ = rotate_x(x_, y_), rotate_y(x_, y_)
#     x = x_ + x_offset
#     y = y_ + y_offset
#
#     rbboxes = np.concatenate([x[..., None], y[..., None]], axis=-1)
#     rbboxes = rbboxes.reshape(-1, 8)
#     return rbboxes
def create_shapes_dataset():
    """
    Creates image detection and segmentation datasets in various formats, according to config files defitions

    :return:
    :rtype:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/dataset_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    image_dir = config['image_dir'] # './dataset/images/{split}/'

    # train val test split sizes:
    splits = config["splits"]
    shapes_config_file = config['shapes_config_file']
    shapes_dataset = ShapesDataset(shapes_config_file)
    output_dir = config['output_dir']

    for split in splits: # loop: train, validation, test
        print(f'create {split} files:')
        nentries = int(splits[split])
        # create dirs for output if missing:
        images_out_dir = f'{output_dir}/{image_dir}/{split}'

        images_out_path = Path(images_out_dir)
        # create image dir for split - if needed
        images_out_path.mkdir(parents=True, exist_ok=True)

        images_filenames, images_sizes, images_bboxes, categories_lists, category_names, category_ids, images_polygons = \
            shapes_dataset.create_dataset(
                nentries,
                f'{images_out_dir}')

        labels_out_dir = Path(f"{output_dir}/{config[f'labels_dir']}/{split}")
        labels_out_dir.mkdir(parents=True, exist_ok=True)
        # related label file has same name with .txt ext - split filename, replace ext to txt:
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
        bbox_entries = create_detection_entries(images_bboxes, images_sizes, categories_lists)


        # 3. text file per image
        if config.get('labels_file_format')=='detection_yolov5':
            # related label file has same name with .txt ext - split filename, replace ext to txt:
            bbox_entries = create_detection_entries(images_bboxes, images_sizes, categories_lists)
            entries_to_files(bbox_entries, label_out_fnames, labels_out_dir)
        elif config.get('labels_file_format') == 'dota_obb':
            rbboxes = []
            for idx, bbox_entry in enumerate(bbox_entries):  # loop on images
                bbox_entries = [[float(idx) for idx in entry.split(' ')] for entry in bbox_entry]
                bbox_entries = np.array(bbox_entries)
                theta=45 # todo ronen debug

                bbox_entries = xywh2xyxy(bbox_entries, theta)

                bbox_entries[:,:8]=rotate(bbox_entries[:,:8].reshape([-1, 4,2]), theta).reshape(-1, 8) # todo new
                rbboxes.append(bbox_entries)

            # labels_out_dir.mkdir(parents=True, exist_ok=True)
            dota_entries_to_files(rbboxes, category_names, label_out_fnames, labels_out_dir)

        elif config.get('labels_file_format')=='kpts_detection_yolov5':
            # related label file has same name with .txt ext - split filename, replace ext to txt:
            kpts_entries=create_detection_kpts_entries(images_bboxes, images_polygons, images_sizes, categories_lists)
            entries_to_files(kpts_entries, label_out_fnames, labels_out_dir)

            # create dataset yaml:
            npkts = len(images_polygons[0][0])  # [nimg,nobj, nvertices, 2], assumed nvertices identical to all (same shapes)
            out_filename = f"{output_dir}/dataset.yaml"
            dataset_yaml = {
                'nc': 1,
                'names': {0: 0},
                'kpt_shape': [npkts, 3],  # x,y,valid
                'skeleton': [],
                'train': f"{config['base_dir']}/{config['image_dir']}/train",
                'val': f"{config['base_dir']}/{config['image_dir']}/valid"
            }
        #  4. Ultralitics like segmentation
        elif config.get('labels_file_format') == 'segmentation_yolov5':
            # related label file has same name with .txt ext - split filename, replace ext to txt:
            create_segmentation_label_files(images_polygons, images_sizes,
                                            categories_lists, label_out_fnames, labels_out_dir)

            # 1. coco format (i.e. dataset entries defined by a json file)
        elif config.get('labels_file_format') == 'detection_coco_json_format':
            labels_out_dir = config['coco_json_labels_file_path'].replace('{split}', split)
            images_filenames = [f'{config["image_dir"]}/{split}/{images_filename}' for images_filename in
                                images_filenames]
            create_coco_json_lable_files(images_filenames, images_sizes, images_bboxes, categories_lists,
                                         category_names, category_ids,
                                         labels_out_dir)
        elif config.get('labels_file_format') == 'detection_unified_textfile':
            labels_out_dir = config['labels_all_entries_file'].replace("{split}", split)
            labels_dir = Path(os.path.dirname(labels_out_dir))
            labels_dir.mkdir(parents=True, exist_ok=True)
            create_detection_labels_unified_file(images_filenames, images_bboxes, categories_lists,
                                                 labels_out_dir)

            with open(out_filename, 'w') as outfile:
                yaml.dump(dataset_yaml, outfile, default_flow_style=False)



    # write category names file:
    print(f'Saving category_names_file {output_dir}/{config["category_names_file"]}')
    with open(f'{output_dir}/{config["category_names_file"]}', 'w') as f:
        for category_name in category_names:
            f.write(f'{category_name}\n')


if __name__ == '__main__':
    create_shapes_dataset()
