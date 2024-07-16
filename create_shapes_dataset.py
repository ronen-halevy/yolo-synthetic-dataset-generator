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

import random
from src.create.utils import  draw_images, write_images_to_file, write_entries_to_files
from src.create.create_detection_labels import CreateDetectionLabels
from src.create.create_segmentation_labels import CreateSegmentationLabels
from src.create.create_obb_labels import create_obb_labels, CreateObbLabels
from src.create.create_kpts_labels import CreatesKptsLabels
from src.create.create_polygons import CreatePolygons


if __name__ == '__main__':

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
    labels_format_type = config['labels_format_type']
    output_dir = f'{config["output_dir"]}'.replace("{labels_format_type}", labels_format_type)
    bg_color = config['bg_color']
    # shapes_dataset = ShapesDataset(config)
    create_polygons = CreatePolygons(config)
    create_detection_labels = CreateDetectionLabels(config['iou_thresh'], config['bbox_margin'])
    create_obb_labels = CreateObbLabels(config['iou_thresh'], config['bbox_margin'])
    create_kpts_labels = CreatesKptsLabels(config['iou_thresh'], config['bbox_margin'])
    create_segmentation_labels = CreateSegmentationLabels()
    categories_names = create_polygons.categories_names

    # create_dataset_config_file:
    dataset_yaml = {
        'path': f'{output_dir}',
        'train': 'images/train',
        'val': 'images/valid',
        'nc': len(categories_names),
        'names': categories_names,
    }
    if labels_format_type == 'kpts':
        nkpts = max(create_polygons.shapes_nvertices)
        dataset_yaml.update({'kpt_shape': [nkpts,3], 'skeleton': []})
    out_filename = f"{output_dir}/dataset.yaml"
    print(f'\n writing {out_filename}')
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    with open(out_filename, 'w') as outfile:
        yaml.dump(dataset_yaml, outfile, default_flow_style=False)

    # create dataset for all splits:
    print(f'\n Creating dataset entries: {splits}')
    for split in splits: # loop: train, validation, test
        nentries = int(splits[split])
        batch_image_size, batch_categories_ids, batch_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas=create_polygons.create_batch_polygons(nentries)

        # create dirs for output if missing:
        images_out_dir = f'{output_dir}/{image_dir}/{split}'
        images_out_path = Path(images_out_dir)
        # create image dir for split - if needed
        images_out_path.mkdir(parents=True, exist_ok=True)
        labels_out_dir = Path(f"{output_dir}/{config[f'labels_dir']}/{split}")
        labels_out_dir.mkdir(parents=True, exist_ok=True)

        config_image_size = config['image_size']
        sel_index = [random.randint(0, len(config['image_size']) - 1) for idx in range(len(batch_polygons))] # randomly select img size index from config
        images_size = tuple(np.array(config_image_size)[sel_index])
        images_filenames = [f'img_{idx:06d}.jpg' for idx in range(len(batch_polygons))]
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]

        # 3. text file per image
        if labels_format_type == 'detection':
            batch_labels = create_detection_labels.run(batch_polygons, images_size, batch_categories_ids)
        elif labels_format_type == 'obb':
            batch_labels, batch_polygons = create_obb_labels.run(batch_polygons, batch_image_size, batch_obb_thetas, batch_categories_names)
        elif labels_format_type=='kpts':
            batch_labels = create_kpts_labels.run(batch_polygons, batch_image_size,  batch_categories_names)
        #  4. Ultralitics like segmentation
        elif labels_format_type == 'segmentation':
            batch_labels=create_segmentation_labels.run(batch_polygons, images_size, batch_categories_ids)
        # save result images and label files:
        print(f'writing {len(label_out_fnames)} label files to {labels_out_dir}')
        write_entries_to_files(batch_labels, label_out_fnames, labels_out_dir)
        images = draw_images(batch_polygons, batch_objects_colors, images_size, bg_color)
        images_out_dir = Path(f"{output_dir}/{config[f'image_dir']}/{split}")
        print(f'writing {len(images_filenames)} image files to {images_out_dir}')
        write_images_to_file(images, images_out_dir,images_filenames)

    # write category names file:
    print(f'Saving category_names_file {output_dir}/{config["category_names_file"]}')
    with open(f'{output_dir}/{config["category_names_file"]}', 'w') as f:
        for category_name in categories_names:
            f.write(f'{category_name}\n')


