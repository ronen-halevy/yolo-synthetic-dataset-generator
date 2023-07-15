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


def gen_label_text_file(images_bboxes, images_filenames, images_objects_categories_names, map_category_id, output_dir, split):
    entries = []
    for filename, categories_names, bboxes in zip(images_filenames, images_objects_categories_names, images_bboxes):
        # im_height = image_entry['height']
        # im_width = image_entry['width']
        # annots = [annot for annot in annotations if annot['image_id'] == image_entry['id']]
        # filename = image_entry['file_name']

        entry = f'{output_dir}/{split}/{filename} '
        for bbox, category_name in zip(bboxes, categories_names):
            # bbox = annot['bbox']
            # category = categories_records[annot['category_id']]['id']
            bbox_arr = np.array(bbox, dtype=float)
            xyxy_bbox = [bbox_arr[0], bbox_arr[1], bbox_arr[0] + bbox_arr[2], bbox_arr[1] + bbox_arr[3]]
            for vertex in xyxy_bbox:
                entry = f'{entry}{vertex},'
            entry = f'{entry}{float(map_category_id[category_name])} '
        entries.append(entry)
        opath = f'{output_dir}/{split}/all_entries.txt'
        file = open(opath, 'w')
        for item in entries:
            file.write(item + "\n")
        file.close()

def fill_categories_records(shapes):
    categories_records = []
    added_category_names = []
    map_categories_id = {}
    id = 0
    for shape in shapes:
        category_name = shape['category_name']
        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": shape['super_category'],
                "id": id,
                "name": category_name,
            })
            added_category_names.append(category_name)

            map_categories_id.update({category_name: id})
            id += 1

    return categories_records, map_categories_id


def create_coco_dataset(images_filenames, images_sizes, images_bboxes, images_objects_categories_names, category_names, super_category_names):

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    map_categories_id = {}

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
            map_categories_id.update({category_name: id})
            id += 1

    images_records=[]
    annotatons_records=[]
    for example_id, (image_filename, image_size, bboxes, objects_categories_names) in enumerate(zip (images_filenames, images_sizes, images_bboxes, images_objects_categories_names)):

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
        for bbox, category_name in zip(bboxes, objects_categories_names):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": map_categories_id[category_name],
                "id": anno_id
            }
            )
            anno_id += 1




    date_today = date.today()
    info = {
        "description": "Shapes Dataset",
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

    return output_records


####

class CreateBbox:
    pass




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


    def run(self, shapes, image_size, min_objects_in_image, max_objects_in_image, bg_color, iou_thresh, margin_from_edge,
                   bbox_margin,
                   size_fluctuation

                   ):
        image = Image.new('RGB', image_size, bg_color)
        # draw = ImageDraw.Draw(image)
        num_of_objects = np.random.randint(min_objects_in_image, max_objects_in_image + 1)
        bboxes = []
        objects_categories_names = []
        for index in range(num_of_objects):

            shape_entry = np.random.choice(shapes)
            try:
                axis_ratio = shape_entry['shape_aspect_ratio']
            except Exception as e:
                print(e)
                pass
            shape_width_choices = shape_entry['shape_width_choices'] if 'shape_width_choices' in shape_entry else 1
            try:
                bbox = self.create_bbox(image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                                   size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(
                    f'Failed in placing the {index} object into image:\n{msg}.\nHere is the failed-to-be-placed shape entry: {shape_entry}')

            if len(bbox):
                bboxes.append(bbox.tolist())
            else:
                break
            objects_categories_names.append(shape_entry['category_name'])

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

        return image, bboxes, objects_categories_names




# # output box format: x_center, y_center, w,h


# prepare a single label text file.  box format: xy_min, xy_max:




anno_id=0





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.yaml',
                        help='yaml config file')

    parser.add_argument("--shapes_file", type=str, default='config/shapes.yaml',
                        help='shapes yaml config file')

    # parser.add_argument("--class_names_out_file", type=str, default='dataset/class.names',
    #                     help='class_names output _file')

    args = parser.parse_args()
    config_file = args.config_file
    shapes_file = args.shapes_file
    # class_names_out_file = args.class_names_out_file

    with open(shapes_file, 'r') as stream:
        shapes = yaml.safe_load(stream)

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config["output_dir"]
    class_names_out_file = f'{output_dir}/{config["class_names_file"]}'
    cb = CreateBbox()

    image_size = config['image_size']
    min_objects_in_image=config['min_objects_in_image']
    max_objects_in_image=config['max_objects_in_image']
    bg_color=tuple(config['bg_color'])
    iou_thresh=config['iou_thresh']
    margin_from_edge=config['margin_from_edge']
    bbox_margin=config['bbox_margin']
    size_fluctuation=config['size_fluctuation']

    categories_records, map_categories_id = fill_categories_records(shapes)


    splits = config["splits"]

    anno_id = 0

    for split in splits:

        split_output_dir = Path(f'{output_dir}/{split}')
        split_output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Creating {split} split in {output_dir}/{split}: {int(splits[split])} examples.\n Running....')

        images_records = []
        annotatons_records = []

        images_filenames = []
        images_sizes = []
        images_bboxes=[]
        images_objects_categories_names=[]


        for example_id in range(int(splits[split])):
            try:
                image, bboxes, objects_categories_names=cb.run(shapes, image_size, min_objects_in_image, max_objects_in_image, bg_color, iou_thresh,
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
            images_objects_categories_names.append(objects_categories_names)

        category_names =  [ shape['category_name'] for shape in shapes]
        super_category_names =  [ shape['super_category'] for shape in shapes]
        map_category_id =  { shape['category_name']: idx for idx, shape in enumerate(shapes)}


        # contributor = config.get('contributor')
        # licenses = config.get('licenses')



        output_records = create_coco_dataset(images_filenames, images_sizes, images_bboxes, images_objects_categories_names, category_names,
                                super_category_names)

        annotations_path = f'{output_dir}/{split}/images/annotations.json'

        print(f'Save annotation  in {annotations_path}')
        with open(annotations_path, 'w') as fp:
            json.dump(output_records, fp)

        # # 2. single text file:
        # # prepare a single label text file.  row format: image file path, x_min, y_min, x_max, y_max, classid
        # gen_label_text_file(annotatons_records, images_records, categories_records, output_dir, split)
        #
        # # 3. Ultralitics like format
        # # prepare a label text file per image.  box format: x_center, y_center, w,h
        # gen_per_image_label_text_file(annotatons_records, images_records, categories_records, f'{output_dir}/{split}/')

        gen_label_text_file(images_bboxes, images_filenames, images_objects_categories_names, map_category_id, output_dir, split)


# try:
    #     create_dataset(config_file, shapes_file)
    # except Exception as e:
    #     print(e)
    #     exit(1)


if __name__ == '__main__':


    main()




