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

import numpy as np
from PIL import Image, ImageDraw
import math
import yaml
import json
from datetime import date, datetime
import random
import argparse
import os
from pathlib import Path
from output_formatters import create_per_image_labels_files, create_row_text_labels_file, create_coco_labels_file


class ShapesDataset:

    def __compute_iou(self, box1, box2):
        """x_min, y_min, x_max, y_max"""
        area_box_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        if y_min >= y_max or x_min >= x_max:
            return 0
        return ((x_max - x_min) * (y_max - y_min)) / (area_box_2 + area_box_1)

    def __create_bbox(self, image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                      size_fluctuation=0.01):
        """

        :param image_size: Canvas size
        :type image_size:
        :param bboxes:
        :type bboxes:
        :param shape_width_choices:
        :type shape_width_choices:
        :param axis_ratio:
        :type axis_ratio:
        :param iou_thresh:
        :type iou_thresh:
        :param margin_from_edge:
        :type margin_from_edge:
        :param size_fluctuation:
        :type size_fluctuation:
        :return:
        :rtype:
        """
        max_count = 10000
        count = 0
        # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
        while True:
            shape_width = np.random.choice(shape_width_choices)
            shape_height = shape_width * axis_ratio * random.uniform(1 - size_fluctuation, 1)
            # add fluctuations - config defuned
            shape_width = shape_width * random.uniform(1 - size_fluctuation, 1)
            radius = np.array([shape_width / 2, shape_height / 2])
            center = np.random.randint(
                low=radius + margin_from_edge, high=np.floor(image_size - radius - margin_from_edge), size=2)
            # bbox_sides = radius
            new_bbox = np.concatenate(np.tile(center, 2).reshape(2, 2) +
                                      np.array([np.negative(radius), radius]))
            # iou new shape bbox with all prev bboxes. skip shape if max iou > thresh - try another placement for shpe
            iou = list(map(lambda x: self.__compute_iou(new_bbox, x), bboxes))

            if len(iou) == 0 or max(iou) <= iou_thresh:
                break
            if count > max_count:
                max_iou = max(iou)
                raise Exception(
                    f'Shape Objects Placement Failed after {count} placement itterations: max(iou)={max_iou}, '
                    f'but required iou_thresh is {iou_thresh} shape_width: {shape_width},'
                    f' shape_height: {shape_height}. . \nHint: reduce objects size or quantity of objects in an image')
            count += 1

        return new_bbox

    def draw_shape(self, draw, shape, bbox, fill_color, outline_color):
        if shape in ['ellipse', 'circle']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.ellipse([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)

        elif shape in ['rectangle', 'square']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.rectangle((x_min, y_min, x_max, y_max), fill=fill_color, outline=outline_color, width=3)

        elif shape == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape in ['trapezoid''hexagon']:
            sides = 5 if shape == 'trapezoid' else 6
            x_min, y_min, x_max, y_max = bbox.tolist()
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            rad_x, rad_y = (x_max - x_min) / 2, (y_max - y_min) / 2
            xy = [
                (math.cos(th) * rad_x + center_x,
                 math.sin(th) * rad_y + center_y)
                for th in [i * (2 * math.pi) / sides for i in range(sides)]
            ]
            draw.polygon(xy, fill=fill_color, outline=outline_color)

    def create_ds_example(self, shapes_attributes, image_size, num_of_objects, bg_color, iou_thresh,
                          margin_from_edge,
                          bbox_margin,
                          size_fluctuation

                          ):
        image = Image.new('RGB', image_size, bg_color)
        draw = ImageDraw.Draw(image)
        bboxes = []
        objects_categories_names = []
        objects_categories_indices = []

        for entry_id, category_name, shape_aspect_ratio, shape_width_choices, fill_color, outline_color in shapes_attributes:
            try:
                bbox = self.__create_bbox(image_size, bboxes, shape_width_choices, shape_aspect_ratio, iou_thresh,
                                          margin_from_edge,
                                          size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(f'Failed in __create_bbox :\n{msg}.\nHere is the failed-to-be-placed shape entry: {entry_id}, {category_name}')

            if len(bbox):
                bboxes.append(bbox.tolist())
            else:
                break
            self.draw_shape(draw, category_name, bbox, fill_color, outline_color)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(entry_id)

        bboxes = np.array(bboxes)
        # transfer bbox coordinate to:  [xmin, ymin, w, h]: (bbox_margin is added distance between shape and bbox)
        bboxes = [bboxes[:, 0] - bbox_margin,
                  bboxes[:, 1] - bbox_margin,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]  # / np.tile(image_size,2)

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return image, bboxes, objects_categories_indices, objects_categories_names

    def create_dataset(self, shapes, image_size,
                       min_objects_in_image,
                       max_objects_in_image, bg_color,
                       iou_thresh,
                       margin_from_edge,
                       bbox_margin,
                       size_fluctuation, nentries, output_dir):
        """

        :param shapes:
        :param image_size:
        :param min_objects_in_image:
        :param max_objects_in_image:
        :param bg_color:
        :param iou_thresh:
        :param margin_from_edge:
        :param bbox_margin:
        :param size_fluctuation:
        :param nentries:
        :param output_dir:
        :return:
        """

        images_filenames = []
        images_sizes = []
        images_bboxes = []
        images_objects_categories_indices = []
        images_objects_categories_names = []
        for example_id in range(nentries):
            num_of_objects = np.random.randint(min_objects_in_image, max_objects_in_image + 1)

            shape_entris= [np.random.choice(shapes) for idx in range(num_of_objects)]
            shapes_attributes = [[shape_entry['id'],  shape_entry['category_name'], shape_entry['shape_aspect_ratio'], shape_entry['shape_width_choices'],
                                 tuple(shape_entry['fill_color']), tuple(shape_entry['outline_color'])] for shape_entry in shape_entris]
            try:
                image, bboxes, objects_categories_indices, objects_categories_names = self.create_ds_example(shapes_attributes,
                                                                                                             image_size,
                                                                                                             num_of_objects,
                                                                                                             bg_color,
                                                                                                             iou_thresh,
                                                                                                             margin_from_edge,
                                                                                                             bbox_margin,
                                                                                                             size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(f'Error: While creating the {example_id}th image: {msg}')
            image_filename = f'img_{example_id + 1:06d}.jpg'
            file_path = f'{output_dir}/images/{image_filename}'
            image.save(file_path)
            if len(bboxes) == 0:
                continue

            images_filenames.append(image_filename)
            images_sizes.append([image.height, image.width])
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)

        return images_filenames, images_sizes, images_bboxes, images_objects_categories_indices


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
                                    f'{output_dir}/{split}/')


if __name__ == '__main__':
    main()
