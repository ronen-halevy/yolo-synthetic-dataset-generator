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

# todo - move to utils:
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

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

    image_dir = './dataset/images/{split}/'

    # train val test split sizes:
    splits = config["splits"]
    shapes_config_file = config['shapes_config_file']
    shapes_dataset = ShapesDataset(shapes_config_file)
    # return # debug!!!
    output_dir = config['test_output_dir']
    print('\nrendering dataset images with bbox and mask overlays\n')
    output_dir=increment_path(output_dir)
    base_dir = config['output_dir']

    for split in splits:
        print(f'create {split} files:')
        nentries = int(splits[split])
        # create dirs for output if missing:
        images_out_dir = Path(f'{image_dir.replace("{split}", split)}')
        # create image dir for split - if needed
        images_out_dir.mkdir(parents=True, exist_ok=True)

        images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, category_names, category_ids, images_polygons = \
            shapes_dataset.create_dataset(
                nentries,
                f'{images_out_dir}')

        # 1. coco format (i.e. dataset entries defined by a json file)
        if config.get('labels_file_format')=='detection_coco_json_format':
            annotations_output_path = './dataset/images/{split}/annotations.json'.replace('{split}', split)
            images_filenames = [f'{config["image_dir"].replace("{split}", split)}{images_filename}' for images_filename in images_filenames]
            create_coco_json_lable_files(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices,
                           category_names, category_ids,
                           annotations_output_path)
        if config.get('labels_file_format') == 'detection_label_unified_file_path':
            labels_out_path = f'./{base_dir}/{split}/all_entries.txt'

            labels_split__out_path = labels_out_dir.replace("{split}", split)
            labels_dir = os.path.dirname(labels_split__out_path)
            labels_dir = Path(labels_dir)
            labels_dir.mkdir(parents=True, exist_ok=True)
            create_detection_labels_unified_file(images_filenames, images_bboxes, images_objects_categories_indices,
                                    labels_split__out_path)

        # 3. text file per image
        if config.get('labels_file_format')=='detection_yolov5':
            labels_out_dir = Path('./dataset/labels/{split}'.replace("{split}", split))
            labels_out_dir.mkdir(parents=True, exist_ok=True)
            create_detection_lable_files(images_filenames, images_bboxes, images_sizes,
                                        images_objects_categories_indices
                                        , labels_out_dir)
        #  4. Ultralitics like segmentation
        if config.get('labels_file_format')=='segmentation_yolov5':
            labels_out_dir='./ dataset/labels/{split}'
            labels_out_dir = Path(labels_out_dir.replace("{split}", split))
            Path(labels_out_dir).mkdir(parents=True, exist_ok=True)
            create_segmentation_label_files(images_filenames, images_polygons, images_sizes,
                                          images_objects_categories_indices,
                                          labels_out_dir)
        print(f'rendering results image and labels overlays: {output_dir}/{split}\n')
        render.main(images_out_dir, labels_out_dir, f'{output_dir}/{split}', category_names)


    # write category names file:
    print(f'Saving {config["category_names_file"]}')
    with open(config['category_names_file'], 'w') as f:
        for category_name in category_names:
            f.write(f'{category_name}\n')
    # render dataset for demo and verification:




if __name__ == '__main__':
    create_shapes_dataset()
