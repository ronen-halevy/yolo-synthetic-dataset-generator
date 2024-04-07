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

    for split in splits: # loop: train, validation, test
        print(f'create {split} files:')
        nentries = int(splits[split])
        # create dirs for output if missing:
        images_out_dir = f'{image_dir.replace("{split}", split)}'
        images_out_path = Path(images_out_dir)
        # create image dir for split - if needed
        images_out_path.mkdir(parents=True, exist_ok=True)

        images_filenames, images_sizes, images_bboxes, categories_lists, category_names, category_ids, images_polygons = \
            shapes_dataset.create_dataset(
                nentries,
                f'{images_out_dir}')

        # 1. coco format (i.e. dataset entries defined by a json file)
        if config.get('labels_file_format')=='detection_coco_json_format':
            labels_out_dir = config['coco_json_labels_file_path'].replace('{split}', split)
            images_filenames = [f'{config["image_dir"].replace("{split}", split)}{images_filename}' for images_filename in images_filenames]
            create_coco_json_lable_files(images_filenames, images_sizes, images_bboxes, categories_lists,
                           category_names, category_ids,
                           labels_out_dir)
        elif config.get('labels_file_format') == 'detection_unified_textfile':
            labels_out_dir = config['labels_all_entries_file'].replace("{split}", split)
            labels_dir = Path(os.path.dirname(labels_out_dir))
            labels_dir.mkdir(parents=True, exist_ok=True)
            create_detection_labels_unified_file(images_filenames, images_bboxes, categories_lists,
                                    labels_out_dir)

        # 3. text file per image
        elif config.get('labels_file_format')=='detection_yolov5':
            labels_out_dir = Path(config['labels_dir'].replace("{split}", split))
            labels_out_dir.mkdir(parents=True, exist_ok=True)
            # related label file has same name with .txt ext - split filename, replace ext to txt:
            label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
            create_detection_lable_files(images_bboxes, images_sizes,
                                        categories_lists, label_out_fnames, labels_out_dir)

        #  4. Ultralitics like segmentation
        elif config.get('labels_file_format')=='segmentation_yolov5':
            labels_out_dir = Path(config['labels_dir'].replace("{split}", split))
            Path(labels_out_dir).mkdir(parents=True, exist_ok=True)
            # related label file has same name with .txt ext - split filename, replace ext to txt:
            label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
            create_segmentation_label_files(images_polygons, images_sizes,
                                          categories_lists, label_out_fnames, labels_out_dir)

    # write category names file:
    print(f'Saving {config["category_names_file"]}')
    with open(config['category_names_file'], 'w') as f:
        for category_name in category_names:
            f.write(f'{category_name}\n')


if __name__ == '__main__':
    create_shapes_dataset()
