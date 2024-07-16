#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : main_create_polygons_dataset.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#   main method for  polygons dataset creation.
#   1. reads splits sizes and destination output path from config.json
#   2. Creates an instance of  ShapesDataset and generates the dataset
#   3. Envokes formatters to save dataset labels in various formats e.g. coco, multi text file (yolov5 ultralics like),
#   single text file
# ================================================================
import os
import yaml
import argparse
from pathlib import Path

import random
from src.create.utils import  draw_images, write_images_to_file, write_entries_to_files
from src.create.create_detection_labels import CreateDetectionEntries
from src.create.create_segmentation_labels import CreateSegmentationEntries
from src.create.create_obb_labels import CreateObbEntries
from src.create.create_kpts_labels import CreatesKptsEntries


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
    # polygons_config_file = config['polygons_config_table']
    labels_format_type = config['labels_format_type']
    output_dir = f'{config["output_dir"]}'.replace("{labels_format_type}", labels_format_type)
    bg_color = config['bg_color']

    if labels_format_type == 'detection':
        create_dataset = CreateDetectionEntries(config, config['iou_thresh'], config['bbox_margin'])
    elif labels_format_type == 'obb':
        create_dataset = CreateObbEntries(config, config['iou_thresh'], config['bbox_margin'])
    elif labels_format_type == 'kpts':
        create_dataset = CreatesKptsEntries(config,config['iou_thresh'], config['bbox_margin'])
    elif labels_format_type == 'segmentation':
        create_dataset = CreateSegmentationEntries(config)
    # categories_names = create_polygons.categories_names
    categories_names=create_dataset.categories_names
    # create_dataset_config_file:
    dataset_yaml = {
        'path': f'{output_dir}',
        'train': 'images/train',
        'val': 'images/valid',
        'nc': len(categories_names),
        'names': categories_names,
    }
    if labels_format_type == 'kpts':
        nkpts = max(create_dataset.polygons_nvertices)
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
        batch_polygons, batch_labels, batch_objects_colors, batch_image_size = create_dataset.run(nentries)

        # arrange images and labels output dirs per split
        images_out_dir = f'{output_dir}/{image_dir}/{split}'
        images_out_path = Path(images_out_dir)
        images_out_path.mkdir(parents=True, exist_ok=True)
        labels_out_dir = Path(f"{output_dir}/{config[f'labels_dir']}/{split}")
        labels_out_dir.mkdir(parents=True, exist_ok=True)
        # arrange filenames:
        images_filenames = [f'img_{idx:06d}.jpg' for idx in range(len(batch_polygons))]
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
        # save label files"
        print(f'writing {len(label_out_fnames)} label files to {labels_out_dir}')
        write_entries_to_files(batch_labels, label_out_fnames, labels_out_dir)
        # draw images and save to files"
        images = draw_images(batch_polygons, batch_objects_colors, batch_image_size, bg_color)
        images_out_dir = Path(f"{output_dir}/{config[f'image_dir']}/{split}")
        print(f'writing {len(images_filenames)} image files to {images_out_dir}')
        write_images_to_file(images, images_out_dir,images_filenames)

    # write category names file:
    print(f'Saving category_names_file {output_dir}/{config["category_names_file"]}')
    with open(f'{output_dir}/{config["category_names_file"]}', 'w') as f:
        for category_name in categories_names:
            f.write(f'{category_name}\n')


