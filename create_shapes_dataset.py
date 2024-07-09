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
                                    normalize_bboxes, entries_to_files, dota_entries_to_files,arrange_segmentation_entries, create_obb_entries, rotate_obb_bbox_entries, rotate_polygon_entries, remove_dropped_bboxes, create_detection_kpts_entries,
                                    create_detection_labels_unified_file, write_images_to_file)
from src.shapes_dataset import ShapesDataset
from src.create_label_files import rotate

import math



def draw_images(images_polygons, images_objects_colors=None, images_size=None, bg_color_set=['red']):
    # related label file has same name with .txt ext - split filename, replace ext to txt:

    images_filenames = []
    images = []
    # images_objects_colors = np.full(list(np.array(images_polygons).shape[0:2])+[1], ['blue'])
    # images_size = np.full([len(images_polygons),2],(640,640))

    # for idx, (bboxes, categories_list, category_names, category_ids, image_polygons, image_objects_colors) in enumerate(zip(images_bboxes, categories_lists, categories_names, categories_ids, images_polygons, images_objects_colors)):
    for idx, (image_polygons, image_objects_color, image_size) in enumerate(
            zip(images_polygons, images_objects_colors, images_size)):

        # save image files
        # sel_index = random.randint(0, len(image_size)-1) # randomly select img size index from config
        # image_size= image_size[sel_index]
        bg_color = np.random.choice(bg_color_set)
        image = Image.new('RGB', tuple(image_size), bg_color)
        draw = ImageDraw.Draw(image)

        for image_polygon, image_object_color in zip(image_polygons, image_objects_color):
            color = np.random.choice(image_object_color)
            draw.polygon(image_polygon.flatten().tolist(), fill=ImageColor.getrgb(color))
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
        batch_bboxes, categories_lists, categories_names, categories_ids, batch_polygons, images_objects_colors, obb_thetas = \
            shapes_dataset.create_images_shapes(
                nentries)



        config_image_size = config['image_size']
        sel_index = [random.randint(0, len(config['image_size']) - 1) for idx in range(len(batch_polygons))] # randomly select img size index from config
        images_size = tuple(np.array(config_image_size)[sel_index])
        images_filenames = [f'img_{idx:06d}.jpg' for idx in range(len(batch_polygons))]
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
        bbox_entries = normalize_bboxes(batch_bboxes, images_size, categories_lists)

            # 3. text file per image
        if config.get('labels_file_format')=='detection_yolov5':
                entries_to_files(bbox_entries, label_out_fnames, labels_out_dir)
        elif config.get('labels_file_format') == 'obb':
            batch_polygons, batch_obb_thetas, dropped_ids= rotate_polygon_entries(batch_polygons, images_size, obb_thetas)

            bbox_entries, batch_obb_thetas = remove_dropped_bboxes(bbox_entries, batch_obb_thetas, dropped_ids)

            bbox_entries = create_obb_entries(bbox_entries)
            batch_rbboxes= rotate_obb_bbox_entries(bbox_entries, images_size, batch_obb_thetas)
            entries_to_files(batch_rbboxes, label_out_fnames, labels_out_dir)
        elif config.get('labels_file_format')=='kpts_detection_yolov5':
                # related label file has same name with .txt ext - split filename, replace ext to txt:
                kpts_entries=create_detection_kpts_entries(batch_bboxes, batch_polygons, images_size, categories_lists)
                entries_to_files(kpts_entries, label_out_fnames, labels_out_dir)

                # create dataset yaml:
                npkts = len(batch_polygons[0][0])  # [nimg,nobj, nvertices, 2], assumed nvertices identical to all (same shapes)
                out_filename = f"{output_dir}/dataset.yaml"
                dataset_yaml = {
                    'nc': 1,
                    'names': {0: 0},
                    'kpt_shape': [npkts, 3],  # x,y,valid
                    'skeleton': [],
                    'train': f"{config['base_dir']}/{config['image_dir']}/train",
                    'val': f"{config['base_dir']}/{config['image_dir']}/valid"
                }
                with open(out_filename, 'w') as outfile:
                    yaml.dump(dataset_yaml, outfile, default_flow_style=False)
            #  4. Ultralitics like segmentation
        elif config.get('labels_file_format') == 'segmentation_yolov5':
                # related label file has same name with .txt ext - split filename, replace ext to txt:

                batch_entries = arrange_segmentation_entries(batch_polygons, images_size, categories_lists)
                entries_to_files(batch_entries, label_out_fnames, labels_out_dir)

                # 1. coco format (i.e. dataset entries defined by a json file)
        elif config.get('labels_file_format') == 'detection_coco_json_format':
                labels_out_dir = config['coco_json_labels_file_path'].replace('{split}', split)
                images_filenames = [f'{config["image_dir"]}/{split}/{images_filename}' for images_filename in
                                    images_filenames]
                create_coco_json_lable_files(images_filenames, images_size, batch_bboxes, categories_lists,
                                             categories_names, categories_ids,
                                             labels_out_dir)
        elif config.get('labels_file_format') == 'detection_unified_textfile':
                labels_out_dir = config['labels_all_entries_file'].replace("{split}", split)
                labels_dir = Path(os.path.dirname(labels_out_dir))
                labels_dir.mkdir(parents=True, exist_ok=True)
                create_detection_labels_unified_file(images_filenames, batch_bboxes, categories_lists,
                                                     labels_out_dir)


        images = draw_images(batch_polygons, images_objects_colors, images_size, bg_color)
        images_out_dir = Path(f"{output_dir}/{config[f'image_dir']}/{split}")

        write_images_to_file(images, images_out_dir,images_filenames)

    # write category names file:
    print(f'Saving category_names_file {output_dir}/{config["category_names_file"]}')
    with open(f'{output_dir}/{config["category_names_file"]}', 'w') as f:
        for category_name in categories_names:
            f.write(f'{category_name}\n')


if __name__ == '__main__':
    create_shapes_dataset()
