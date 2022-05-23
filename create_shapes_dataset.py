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
import json
import os
import copy


def compute_iou(box1, box2):
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


def create_bbox(image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                size_fluctuation=0.01):
    max_count = 10000
    count = 0
    while True:

        import random

        shape_width = np.random.choice(shape_width_choices)
        shape_height = shape_width * axis_ratio * random.uniform(1 - size_fluctuation, 1)
        shape_width = shape_width * random.uniform(1 - size_fluctuation, 1)
        radius = np.array([shape_width / 2, shape_height / 2])
        center = np.random.randint(
            radius + margin_from_edge, [np.floor(image_size - radius - margin_from_edge)], 2)

        bbox_sides = radius
        new_bbox = np.concatenate(np.tile(center, 2).reshape(2, 2) +
                                  np.array([np.negative(radius), radius]))

        iou = [compute_iou(new_bbox, bbox) for bbox in bboxes]
        if len(iou) == 0 or max(iou) == iou_thresh:
            break
        if count > max_count:
            max_iou = max(iou)
            raise Exception(
                f'Shape Objects Placement Failed after {count} placement itterations: max(iou)={max_iou}, '
                f'but required iou_thresh is {iou_thresh} shape_width: {shape_width},'
                f' shape_height: {shape_height}. . \nHint: reduce objects size or quantity of objects in an image')
        count += 1

    return new_bbox


def make_image(shapes, image_size, min_objects_in_image, max_objects_in_image, bg_color, iou_thresh, margin_from_edge,
               bbox_margin,
               size_fluctuation, class_mode

               ):
    image = Image.new('RGB', image_size, tuple(bg_color))
    draw = ImageDraw.Draw(image)
    num_of_objects = np.random.randint(min_objects_in_image, max_objects_in_image + 1)
    bboxes = []
    added_shapes_metadata = []
    for index in range(num_of_objects):

        shape_entry = np.random.choice(shapes)
        try:
            axis_ratio = shape_entry['shape_aspect_ratio']
        except Exception as e:
            print(e)
            pass
        shape_width_choices = shape_entry['shape_width_choices'] if 'shape_width_choices' in shape_entry else 1
        try:
            bbox = create_bbox(image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                               size_fluctuation)

        except Exception as e:
            msg = str(e)
            raise Exception(
                f'Failed in placing the {index} object into image:\n{msg}.\nHere is the failed-to-be-placed shape entry: {shape_entry}')

        if len(bbox):
            bboxes.append(bbox.tolist())
        else:
            break
        fill_color = shape_entry['fill_color']
        outline_color = shape_entry['outline_color']

        if shape_entry['shape'] in ['ellipse', 'circle']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.ellipse([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)

        elif shape_entry['shape'] in ['rectangle', 'square']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.rectangle((x_min, y_min, x_max, y_max), fill=fill_color, outline=outline_color, width=3)

        elif shape_entry['shape'] == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape_entry['shape'] == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape_entry['shape'] in ['trapezoid', 'hexagon']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            sides = shape_entry['sides']
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            rad_x, rad_y = (x_max - x_min) / 2, (y_max - y_min) / 2
            xy = [
                (math.cos(th) * rad_x + center_x,
                 math.sin(th) * rad_y + center_y)
                for th in [i * (2 * math.pi) / sides for i in range(sides)]
            ]
            draw.polygon(xy, fill=fill_color, outline=outline_color)

        metadata_entry = copy.deepcopy(shape_entry)
        metadata_entry.pop('shape_width_choices')

        if class_mode == 'color':
            metadata_entry['label']  = metadata_entry['color']
        elif class_mode == 'color_and_shape':
            metadata_entry['label'] = f"{metadata_entry['color']}_{metadata_entry['shape']}"
        else:
            metadata_entry['label'] = metadata_entry['shape']

        added_shapes_metadata.append(metadata_entry)


    bboxes = [[(box[0] - bbox_margin) / image_size[0], (box[1] - bbox_margin) / image_size[1],
               (box[2] + bbox_margin) / image_size[0], (box[3] + bbox_margin) / image_size[1]] for box in bboxes]
    return image, bboxes, added_shapes_metadata


def create_dataset(config, shapes):
    num_of_examples = config["num_of_examples"]

    images_dir = config["images_dir"]

    annotations_path = config["annotations_path"]

    import json
    annotatons = []

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)  # c

    for example in range(int(num_of_examples)):
        try:
            image, bboxes, added_shapes = make_image(shapes, config['image_size'],
                                                     config['min_objects_in_image'],
                                                     config['max_objects_in_image'],
                                                     config['bg_color'],
                                                     config['iou_thresh'],
                                                     config['margin_from_edge'],
                                                     config['bbox_margin'],
                                                     config['size_fluctuation'],
                                                     config['class_mode']
                                                     )
        except Exception as e:
            msg = str(e)
            raise Exception(f'Error: While creating the {example}th image: {msg}')

        if len(bboxes) == 0:
            continue

        image_filename = f'{example + 1:06d}.jpg'
        file_path = f'{images_dir}{image_filename}'

        image.save(file_path)

        annotatons.append({'bboxes': bboxes, 'objects': added_shapes, 'image_filename': image_filename})

    data = {
        "annotations": annotatons
    }

    with open(annotations_path, 'w') as annotation_file:
        json.dump(data, annotation_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    config_file = 'config/config.json'
    shapes_file = 'config/shapes.json'
    with open(config_file) as f:
        config_data = json.load(f)

    with open(shapes_file) as f:
        shapes_data = json.load(f)['shapes']
    try:
        create_dataset(config=config_data, shapes=shapes_data)
    except Exception as e:
        print(e)
        exit(1)
