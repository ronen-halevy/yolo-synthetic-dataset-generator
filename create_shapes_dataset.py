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

from src.create_label_files import (create_coco_json_lable_files,
                                    create_detection_lable_files,create_segmentation_label_files,
                                    create_detection_labels_unified_file)
from src.shapes_dataset import ShapesDataset

import render # renders dataset images with bbox and mask overlays

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

    image_dir = config["image_dir"]
    # train val test split sizes:
    splits = config["splits"]
    shapes_config_file = config['shapes_config_file']
    shapes_dataset = ShapesDataset(shapes_config_file)

    for split in splits:
        print(f'create {split} files:')
        nentries = int(splits[split])
        # create dirs for output if missing:
        images_out_dir = Path(f'{image_dir.replace("{split}", split)}')
        images_out_dir.mkdir(parents=True, exist_ok=True)

        images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, category_names, category_ids, images_polygons = \
            shapes_dataset.create_dataset(
                nentries,
                f'{images_out_dir}')


        # 1. coco format (i.e. dataset entries defined by a json file)
        if config.get('coco_detection_dataset_labels_path'):
            annotations_output_path = config['coco_detection_dataset_labels_path'].replace('{split}', split)
            images_filenames = [f'{config["image_dir"].replace("{split}", split)}{images_filename}' for images_filename in images_filenames]
            create_coco_json_lable_files(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices,
                           category_names, category_ids,
                           annotations_output_path)

        # 2. single text file:
        if config.get('detection_label_unified_file_path'):
            labels_path = config['detection_label_unified_file_path'].replace("{split}", split)
            labels_dir = os.path.dirname(labels_path)
            labels_dir = Path(labels_dir)
            labels_dir.mkdir(parents=True, exist_ok=True)
            create_detection_labels_unified_file(images_filenames, images_bboxes, images_objects_categories_indices,
                                    labels_path)

        # 3. text file per image
        if config.get('detection_label_text_files_path'):
            labels_out_dir = Path(config['detection_label_text_files_path'].replace("{split}", split))
            labels_out_dir.mkdir(parents=True, exist_ok=True)
            create_detection_lable_files(images_filenames, images_bboxes, images_sizes,
                                        images_objects_categories_indices
                                        , labels_out_dir)
        #  4. Ultralitics like segmentation
        if config.get('segmentation_label_files_path'):
            labels_out_dir = Path(config['segmentation_label_files_path'].replace("{split}", split))
            Path(labels_out_dir).mkdir(parents=True, exist_ok=True)
            create_segmentation_label_files(images_filenames, images_polygons, images_sizes,
                                          images_objects_categories_indices,
                                          labels_out_dir)


    # write category names file:
    print(f'Saving {config["category_names_file"]}')
    with open(config['category_names_file'], 'w') as f:
        for category_name in category_names:
            f.write(f'{category_name}\n')
    # render dataset for demo and verification:
    render.main()
if __name__ == '__main__':
    create_shapes_dataset()
