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
from PIL import Image, ImageDraw
from PIL import Image, ImageColor
import random

from src.create_label_files import (create_coco_json_lable_files,
                                    normalize_bboxes, entries_to_files, dota_entries_to_files,create_segmentation_label_files, create_detection_kpts_entries, create_obb_entries,
                                    create_detection_labels_unified_file, write_images_to_file)
from src.shapes_dataset import ShapesDataset



import math


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
def draw_images(images_polygons, images_objects_colors=None, images_size=None, bg_color='white'):
    # related label file has same name with .txt ext - split filename, replace ext to txt:

    images_filenames = []
    images = []
    # images_objects_colors = np.full(list(np.array(images_polygons).shape[0:2])+[1], ['blue'])
    # images_size = np.full([len(images_polygons),2],(640,640))

    # for idx, (bboxes, categories_list, category_names, category_ids, image_polygons, image_objects_colors) in enumerate(zip(images_bboxes, categories_lists, categories_names, categories_ids, images_polygons, images_objects_colors)):
    for idx, (image_polygons, image_objects_colors, image_size) in enumerate(
            zip(images_polygons, images_objects_colors, images_size)):

        # save image files
        # sel_index = random.randint(0, len(image_size)-1) # randomly select img size index from config
        # image_size= image_size[sel_index]
        bg_color = np.random.choice(bg_color)
        image = Image.new('RGB', tuple(image_size), bg_color)
        draw = ImageDraw.Draw(image)

        # draw shape on image:
        # sel_color = np.random.choice(color)
        # draw.polygon(image_polygons, fill=ImageColor.getrgb(sel_color) )
        # draw.polygon(image_polygon, fill=ImageColor.getrgb('red') )

        for image_polygon in image_polygons:
            sel_color = np.random.choice(image_objects_colors[idx])
            draw.polygon(image_polygon.flatten().tolist(), fill=ImageColor.getrgb(sel_color))
        images.append(image)
    return images


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
    shapes_config_file = config['shapes_config']
    shapes_dataset = ShapesDataset(config)
    output_dir = config['output_dir']
    bg_color = config['bg_color']

    for split in splits: # loop: train, validation, test
        print(f'create {split} files:')
        nentries = int(splits[split])
        # create dirs for output if missing:
        images_out_dir = f'{output_dir}/{image_dir}/{split}'

        images_out_path = Path(images_out_dir)
        # create image dir for split - if needed
        images_out_path.mkdir(parents=True, exist_ok=True)

        labels_out_dir = Path(f"{output_dir}/{config[f'labels_dir']}/{split}")
        labels_out_dir.mkdir(parents=True, exist_ok=True)
        images_bboxes, categories_lists, categories_names, categories_ids, images_polygons, images_objects_colors = \
            shapes_dataset.create_dataset(
                nentries)



        config_image_size = config['image_size']
        sel_index = [random.randint(0, len(config['image_size']) - 1) for idx in range(len(images_polygons))] # randomly select img size index from config
        images_size = tuple(np.array(config_image_size)[sel_index])
        images_filenames = [f'img_{idx:06d}.jpg' for idx in range(len(images_polygons))]
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
        bbox_entries = normalize_bboxes(images_bboxes, images_size, categories_lists)

            # 3. text file per image
        if config.get('labels_file_format')=='detection_yolov5':
                entries_to_files(bbox_entries, label_out_fnames, labels_out_dir)
        elif config.get('labels_file_format') == 'dota_obb':
                theta = config['obb_rotate'] #rotate(polygon, theta0)
                from src.create_label_files import rotate

                hbboxes = create_obb_entries(bbox_entries)
                hbboxes[..., :8] = rotate(hbboxes[..., :8].reshape([hbboxes.shape[0],-1, 4, 2]), theta).reshape(-1,hbboxes.shape[1],
                                                                                                    8)  # todo new
                images_polygons = rotate(images_polygons, theta)
                images_polygons = tuple(map(tuple, images_polygons))
                dota_entries_to_files(hbboxes, categories_names, label_out_fnames, labels_out_dir)

        elif config.get('labels_file_format')=='kpts_detection_yolov5':
                # related label file has same name with .txt ext - split filename, replace ext to txt:
                kpts_entries=create_detection_kpts_entries(images_bboxes, images_polygons, images_size, categories_lists)
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
                create_segmentation_label_files(images_polygons, images_size,
                                                categories_lists, label_out_fnames, labels_out_dir)

                # 1. coco format (i.e. dataset entries defined by a json file)
        elif config.get('labels_file_format') == 'detection_coco_json_format':
                labels_out_dir = config['coco_json_labels_file_path'].replace('{split}', split)
                images_filenames = [f'{config["image_dir"]}/{split}/{images_filename}' for images_filename in
                                    images_filenames]
                create_coco_json_lable_files(images_filenames, images_size, images_bboxes, categories_lists,
                                             categories_names, categories_ids,
                                             labels_out_dir)
        elif config.get('labels_file_format') == 'detection_unified_textfile':
                labels_out_dir = config['labels_all_entries_file'].replace("{split}", split)
                labels_dir = Path(os.path.dirname(labels_out_dir))
                labels_dir.mkdir(parents=True, exist_ok=True)
                create_detection_labels_unified_file(images_filenames, images_bboxes, categories_lists,
                                                     labels_out_dir)

                with open(out_filename, 'w') as outfile:
                    yaml.dump(dataset_yaml, outfile, default_flow_style=False)

        images = draw_images(images_polygons, images_objects_colors, images_size, bg_color)
        images_out_dir = Path(f"{output_dir}/{config[f'image_dir']}/{split}")

        write_images_to_file(images, images_out_dir,images_filenames)

    # write category names file:
    print(f'Saving category_names_file {output_dir}/{config["category_names_file"]}')
    with open(f'{output_dir}/{config["category_names_file"]}', 'w') as f:
        for category_name in categories_names:
            f.write(f'{category_name}\n')


if __name__ == '__main__':
    create_shapes_dataset()
