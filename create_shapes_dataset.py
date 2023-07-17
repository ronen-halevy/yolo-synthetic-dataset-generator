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

import yaml
import argparse
from pathlib import Path
from output_formatters import create_per_image_labels_files, create_row_text_labels_file, create_coco_labels_file
from shapes_dataset import ShapesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.yaml',
                        help='yaml config file')

    parser.add_argument("--shapes_file", type=str, default='config/shapes.yaml',
                        help='shapes yaml config file')

    args = parser.parse_args()
    config_file = args.config_file
    shapes_file = args.shapes_file

    with open(shapes_file, 'r') as stream:
        shapes = yaml.safe_load(stream)

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config["output_dir"]
    image_size = config['image_size']
    min_objects_in_image = config['min_objects_in_image']
    max_objects_in_image = config['max_objects_in_image']
    bg_color = tuple(config['bg_color'])
    iou_thresh = config['iou_thresh']
    margin_from_edge = config['margin_from_edge']
    bbox_margin = config['bbox_margin']
    size_fluctuation = config['size_fluctuation']

    splits = config["splits"]
    class_names_out_file = f'{output_dir}/{config["class_names_file"]}'

    for split in splits:
        nentries = int(splits[split])
        # create dirs for output if missing:
        split_output_dir = Path(f'{output_dir}/{split}/images')
        split_output_dir.mkdir(parents=True, exist_ok=True)
        split_output_dir = Path(f'{output_dir}/{split}/labels')
        split_output_dir.mkdir(parents=True, exist_ok=True)

        images_filenames, images_sizes, images_bboxes, images_objects_categories_indices = ShapesDataset().create_dataset(shapes,
                                                                                                             image_size,
                                                                                                             min_objects_in_image,
                                                                                                             max_objects_in_image,
                                                                                                             bg_color,
                                                                                                             iou_thresh,
                                                                                                             margin_from_edge,
                                                                                                             bbox_margin,
                                                                                                             size_fluctuation,
                                                                                                             nentries,
                                                                                                             f'{output_dir}/{split}')

        category_names = [shape['category_name'] for shape in shapes]
        with open(class_names_out_file, 'w') as f:
            for category_name in category_names:
                f.write(f'{category_name}\n')

        super_category_names = [shape['super_category'] for shape in shapes]

        annotations_output_path = f'{output_dir}/{split}/images/annotations.json'

        images_filenames = [f'dataset/{split}/images/{images_filename}' for images_filename in images_filenames]

        # coco format
        create_coco_labels_file(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices,
                                category_names,
                                super_category_names, annotations_output_path)

        # 2. single text file:
        create_per_image_labels_files(images_filenames, images_bboxes, images_sizes,
                                      images_objects_categories_indices
                                      , f'{output_dir}/{split}/')

        # # 3. Ultralitics like format
        create_row_text_labels_file(images_filenames, images_bboxes, images_objects_categories_indices,
                                    f'{output_dir}/{split}')


if __name__ == '__main__':
    main()
