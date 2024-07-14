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

from src.create_label_files import (normalize_bboxes, write_entries_to_files,
                                    create_detection_labels_unified_file, write_images_to_file)
from src.segmentation_labels_utils import arrange_segmentation_entries
from src.kpts_detection_labels_utils import create_detection_kpts_entries
from src.obb_labels_utils import create_obb_entries, rotate_obb_bbox_entries, rotate_polygon_entries, remove_dropped_bboxes, remove_dropped_polygons, append_category_field
from src.coco_json_labels_utils import create_coco_json_lable_files
from src.shapes_dataset import ShapesDataset


def draw_images(images_polygons, images_objects_colors=None, images_size=None, bg_color_set=['red']):
    # related label file has same name with .txt ext - split filename, replace ext to txt:
    images = []
    for idx, (image_polygons, image_objects_color, image_size) in enumerate(
            zip(images_polygons, images_objects_colors, images_size)):

        # save image files
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
    format_type = config["labels_format_type"]
    output_dir = f'{config["output_dir"]}'.replace("{labels_format_type}", format_type)
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
        batch_bboxes, categories_lists, batch_objects_categories_names, categories_names, categories_ids, batch_polygons, images_objects_colors, obb_thetas = \
            shapes_dataset.create_images_shapes(
                nentries)
        config_image_size = config['image_size']
        sel_index = [random.randint(0, len(config['image_size']) - 1) for idx in range(len(batch_polygons))] # randomly select img size index from config
        images_size = tuple(np.array(config_image_size)[sel_index])
        images_filenames = [f'img_{idx:06d}.jpg' for idx in range(len(batch_polygons))]
        label_out_fnames = [f"{os.path.basename(filepath).rsplit('.', 1)[0]}.txt" for filepath in images_filenames]
        bbox_entries = normalize_bboxes(batch_bboxes, images_size, categories_lists)

            # 3. text file per image
        if config['labels_format_type'] == 'detection':
                write_entries_to_files(bbox_entries, label_out_fnames, labels_out_dir)
        elif config['labels_format_type'] == 'obb':
            batch_polygons, batch_obb_thetas, dropped_ids= rotate_polygon_entries(batch_polygons, images_size, obb_thetas)
            bbox_entries = remove_dropped_bboxes(bbox_entries, dropped_ids)
            bbox_entries = create_obb_entries(bbox_entries)
            batch_rbboxes, batch_dropped_ids= rotate_obb_bbox_entries(bbox_entries, images_size, batch_obb_thetas)
            batch_polygons = remove_dropped_polygons(batch_polygons, batch_dropped_ids)
            batch_rbboxes = append_category_field(batch_rbboxes, batch_objects_categories_names)

            def entries_list_to_string(batch_rbboxes):
                batch_rbboxes_strings = []
                for img_rbboxes in batch_rbboxes:
                    img_rbboxes = [' '.join(str(x) for x in img_rbboxes[idx]) for idx in range(len(img_rbboxes))]
                    batch_rbboxes_strings.append(img_rbboxes)
                return batch_rbboxes_strings
            batch_rbboxes = entries_list_to_string(batch_rbboxes)

            write_entries_to_files(batch_rbboxes, label_out_fnames, labels_out_dir)

            out_filename = f"{output_dir}/dataset.yaml"
            dataset_yaml = {
                'path': f'{config["base_dir"]}/{output_dir}',
                'train': 'images/train',
                'val': 'images/valid',
                'nc': len(categories_names),
                'names': categories_names,
            }
            with open(out_filename, 'w') as outfile:
                yaml.dump(dataset_yaml, outfile, default_flow_style=False)

        elif config['labels_format_type']=='kpts':
                # related label file has same name with .txt ext - split filename, replace ext to txt:
                kpts_entries=create_detection_kpts_entries(batch_bboxes, batch_polygons, images_size, categories_lists)
                write_entries_to_files(kpts_entries, label_out_fnames, labels_out_dir)
                # create dataset yaml:
                npkts = len(batch_polygons[0][0])  # [nimg,nobj, nvertices, 2], assumed nvertices identical to all (same shapes)
                out_filename = f'{config["base_dir"]}/{output_dir}/dataset.yaml'
                dataset_yaml = {
                    'nc': 1,
                    'names': {0: 0},
                    'kpt_shape': [npkts, 3],  # [npkts, [x,y,valid]]
                    'skeleton': [],
                    'train': f"{config['base_dir']}/{output_dir}/{config['image_dir']}/train",
                    'val': f"{config['base_dir']}/{output_dir}/{config['image_dir']}/valid"
                }
                with open(out_filename, 'w') as outfile:
                    yaml.dump(dataset_yaml, outfile, default_flow_style=False)
            #  4. Ultralitics like segmentation
        elif config['labels_format_type'] == 'segmentation':
                # related label file has same name with .txt ext - split filename, replace ext to txt:
                batch_entries = arrange_segmentation_entries(batch_polygons, images_size, categories_lists)
                write_entries_to_files(batch_entries, label_out_fnames, labels_out_dir)
                # 1. coco format (i.e. dataset entries defined by a json file)
        elif config['labels_format_type'] == 'detection_coco_json_format':
                labels_out_dir = config['coco_json_labels_file_path'].replace('{split}', split)
                images_filenames = [f'{config["image_dir"]}/{split}/{images_filename}' for images_filename in
                                    images_filenames]
                create_coco_json_lable_files(images_filenames, images_size, batch_bboxes, categories_lists,
                                             categories_names, categories_ids,
                                             labels_out_dir)
        elif config['labels_format_type'] == 'detection_unified_textfile':
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
