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


def create_per_image_labels_files(images_filenames, images_bboxes, images_sizes, images_objects_categories_indices,

                                  output_dir):
    """

    :param images_filenames: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_sizes:
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    output_dir = f'{output_dir}labels/'
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # directory already exists
        pass
    for bboxes, filename, images_size, categories_indices in zip(images_bboxes, images_filenames, images_sizes,
                                                              images_objects_categories_indices):
        im_height = images_size[0]
        im_width = images_size[1]

        labels_filename = f"{output_dir}{filename.rsplit('.', 1)[0]}.txt"
        with open(labels_filename, 'w') as f:
            for bbox, category_id in zip(bboxes, categories_indices):
                bbox_arr = np.array(bbox, dtype=float)
                xcycwh_bbox = [(bbox_arr[0] + bbox_arr[2] / 2) / im_width, (bbox_arr[1] + bbox_arr[3] / 2) / im_height,
                               bbox_arr[2] / im_width, bbox_arr[3] / im_height]
                entry = f"{category_id} {' '.join(str(e) for e in xcycwh_bbox)}"
                f.write(entry)


# create a row labels text file. format:
# imag1_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c
# .
# imagm_path x0l,y0l,x0h,y0h,c, ......xnl,ynl,xnh,ynh,c

def create_row_text_labels_file(images_filenames, images_bboxes, images_objects_categories_indices,
                                output_dir):
    """
    :param images_filenames: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    entries = []
    for filename, categories_indices, bboxes in zip(images_filenames, images_objects_categories_indices, images_bboxes):

        image_path = f'{output_dir}/{filename} '
        entry = image_path
        for bbox, category_id in zip(bboxes, categories_indices):
            bbox_arr = np.array(bbox, dtype=float)
            xyxy_bbox = [bbox_arr[0], bbox_arr[1], bbox_arr[0] + bbox_arr[2], bbox_arr[1] + bbox_arr[3]]
            for vertex in xyxy_bbox:
                entry = f'{entry}{vertex},'
            category_id = f'{entry}{float(category_id)} '
        entries.append(category_id)
        opath = f'{output_dir}/all_entries.txt'
        file = open(opath, 'w')
        for item in entries:
            file.write(item + "\n")
        file.close()


# Create a coco like format label file
def create_coco_labels_file(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices,
                            category_names, super_category_names, annotations_output_path):
    """
     :param images_filenames: list of dataset image filenames
    :param images_sizes: list of per image [im.height, im.width] tuples
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices: list of per image categories_indices arrays
    :param category_names: list of all dataset's category names
    :param super_category_names:  list of all dataset's super_category_names
    :param annotations_output_path: path for output file storage
    :return:
    """

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    # map_categories_id = {}

    # fill category
    id = 0
    for category_name, supercategory in zip(category_names, super_category_names):

        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": supercategory,
                "id": id,
                "name": category_name,
            })
            added_category_names.append(category_name)
            # map_categories_id.update({category_name: id})
            id += 1

    images_records = []
    annotatons_records = []
    for example_id, (image_filename, image_size, bboxes, objects_categories_indices) in enumerate(
            zip(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices)):

        # images records:

        images_records.append({
            "license": '',
            "file_name": image_filename,
            "coco_url": "",
            'width': image_size[1],
            'height': image_size[0],
            "date_captured": str(datetime.now()),
            "flickr_url": "",
            "id": example_id
        })

        # annotatons_records
        for bbox, category_id in zip(bboxes, objects_categories_indices):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": category_id,
                "id": anno_id
            }
            )
            anno_id += 1
    date_today = date.today()
    info = {
        "description": " Dataset",
        "url": '',
        # "version": config.get('version', 1.0),
        "year": date_today.year,
        "contributor": '',
        "date_created": str(date_today),
        "licenses": '',
        "categories": categories_records
    }
    output_records = {
        "info": info,
        "licenses": [],
        "images": images_records,
        "categories": categories_records,
        "annotations": annotatons_records
    }
    print(f'Save annotation  in {annotations_output_path}')
    with open(annotations_output_path, 'w') as fp:
        json.dump(output_records, fp)


class ShapesDataset:

    def compute_iou(self, box1, box2):
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

    def create_bbox(self, image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
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
            iou = list(map(lambda x: self.compute_iou(new_bbox, x), bboxes))

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

    def run(self, shapes, image_size, min_objects_in_image, max_objects_in_image, bg_color, iou_thresh,
            margin_from_edge,
            bbox_margin,
            size_fluctuation

            ):
        image = Image.new('RGB', image_size, bg_color)
        # draw = ImageDraw.Draw(image)
        num_of_objects = np.random.randint(min_objects_in_image, max_objects_in_image + 1)
        bboxes = []
        objects_categories_names = []
        objects_categories_indices = []

        for index in range(num_of_objects):
            try:
                shape_entry = np.random.choice(shapes)
                objects_categories_indices.append(shape_entry['id'])
                objects_categories_names.append(shape_entry['category_name'])
                axis_ratio = shape_entry['shape_aspect_ratio']
                shape_width_choices = shape_entry['shape_width_choices'] if 'shape_width_choices' in shape_entry else 1

            except Exception as e:
                print(e)
                pass
            try:
                bbox = self.create_bbox(image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh,
                                        margin_from_edge,
                                        size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(
                    f'Failed in placing the {index} object into image:\n{msg}.\nHere is the failed-to-be-placed shape entry: {shape_entry}')

            if len(bbox):
                bboxes.append(bbox.tolist())
            else:
                break
            # objects_categories_names.append(category_name)

            # fill_color = tuple(shape_entry['fill_color'])
            # outline_color = tuple(shape_entry['outline_color'])
            # draw_shape(draw, shape_entry['shape'], bbox, fill_color, outline_color)

        bboxes = np.array(bboxes)
        # transfer bbox coordinate to:  [xmin, ymin, w, h]: (bbox_margin is added distance between shape and bbox)
        bboxes = [bboxes[:, 0] - bbox_margin,
                  bboxes[:, 1] - bbox_margin,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]  # / np.tile(image_size,2)

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return image, bboxes, objects_categories_indices, objects_categories_names


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
    class_names_out_file = f'{output_dir}/{config["class_names_file"]}'
    cb = ShapesDataset()

    image_size = config['image_size']
    min_objects_in_image = config['min_objects_in_image']
    max_objects_in_image = config['max_objects_in_image']
    bg_color = tuple(config['bg_color'])
    iou_thresh = config['iou_thresh']
    margin_from_edge = config['margin_from_edge']
    bbox_margin = config['bbox_margin']
    size_fluctuation = config['size_fluctuation']
    splits = config["splits"]

    for split in splits:

        split_output_dir = Path(f'{output_dir}/{split}')
        split_output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Creating {split} split in {output_dir}/{split}: {int(splits[split])} examples.\n Running....')

        images_filenames = []
        images_sizes = []
        images_bboxes = []
        images_objects_categories_indices=[]
        images_objects_categories_names = []

        for example_id in range(int(splits[split])):
            try:
                image, bboxes, objects_categories_indices, objects_categories_names = cb.run(shapes, image_size, min_objects_in_image,
                                                                 max_objects_in_image, bg_color, iou_thresh,
                                                                 margin_from_edge,
                                                                 bbox_margin,
                                                                 size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(f'Error: While creating the {example_id}th image: {msg}')
            image_filename = f'img_{example_id + 1:06d}.jpg'
            file_path = f'{output_dir}/{split}/images/{image_filename}'
            image.save(file_path)
            if len(bboxes) == 0:
                continue

            images_filenames.append(image_filename)
            images_sizes.append([image.height, image.width])
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)


        category_names = [shape['category_name'] for shape in shapes]
        super_category_names = [shape['super_category'] for shape in shapes]

        annotations_output_path = f'{output_dir}/{split}/images/annotations.json'

        create_coco_labels_file(images_filenames, images_sizes, images_bboxes, images_objects_categories_indices,
                                category_names,
                                super_category_names, annotations_output_path)


        # # 2. single text file:
        # # prepare a single label text file.  row format: image file path, x_min, y_min, x_max, y_max, classid
        # gen_label_text_file(annotatons_records, images_records, categories_records, output_dir, split)
        #
        # # 3. Ultralitics like format
        # # prepare a label text file per image.  box format: x_center, y_center, w,h

        create_per_image_labels_files(images_filenames, images_bboxes, images_sizes,
                                      images_objects_categories_indices
                                      , f'{output_dir}/{split}/')

        create_row_text_labels_file(images_filenames, images_bboxes, images_objects_categories_indices,
                                    f'{output_dir}/{split}/')


if __name__ == '__main__':
    main()
