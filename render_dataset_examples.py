#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : render_dataset_examples.py
#   Author      : ronen halevy 
#   Created date:  4/25/22
#   Description :
#
# ================================================================

import yaml

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_margin_factor = 0.05

    left, top, right, bottom = zip(*[font.getbbox(display_str) for display_str in display_str_list])
    text_heights= tuple(map(lambda i, j: i - j, bottom, top))
    text_widths = tuple(map(lambda i, j: i - j, right, left))

    text_margins = np.ceil(text_margin_factor * np.array(text_heights))
    text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

    for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
            zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):

        left, top, right, bottom = font.getbbox(display_str)
        text_height = bottom - top
        text_width = right - left

        text_margin = np.ceil(text_margin_factor * text_height)

        draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                        (xmint + text_width + text_margin, text_bottom)),
                       fill=color)

        draw.text((xmint + text_margin, text_bottom - text_height - 3 * text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


def draw_bounding_box(image, boxes, color, thickness=1):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, w, h = box
        draw.line([(xmin, ymin), (xmin, ymin+h), (xmin+w, ymin+h), (xmin+w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image

config_file_path = 'config/config.yaml'

def main():
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    annotations_path = config["annotations_path"]
    images_dir = config['images_dir']

    with open(annotations_path) as file:
        annotations = yaml.safe_load(file)

    plot_setup_params = {
        'num_of_images': 2,
        'start_index': 0,
        'random_select': True,
        'figsize': (7, 7)
    }

    num_of_images = plot_setup_params['num_of_images']
    start_index = plot_setup_params['start_index']
    random_select = plot_setup_params['random_select']

    for idx in range(num_of_images):
        image_index = np.random.randint(start_index, len(annotations)) if random_select else start_index + idx
        image_record = annotations['images'][image_index]
        annotation_records = [annotation for  annotation in annotations['annotations'] if annotation['image_id'] == image_record['id']]
        image_path = f'{images_dir}{image_record["file_name"]}'
        image = Image.open(image_path)
        colors = list(ImageColor.colormap.values())
        color = colors[0]
        bboxes = [annotation_record['bbox'] for annotation_record in annotation_records]
        bboxes = np.array(bboxes)
        bboxes[..., 0:3:2] = bboxes[..., 0:3:2]
        bboxes[..., 1:4:2] = bboxes[..., 1:4:2]
        annotated_bbox_image = draw_bounding_box(image, bboxes, color, thickness=1)
        category_ids = [annotation_record['category_id'] for annotation_record in annotation_records]

        category_names = [category['name'] for category in annotations['categories'] if category['id'] in category_ids]


        annotated_text_image = draw_text_on_bounding_box(annotated_bbox_image, bboxes[..., 1], bboxes[..., 0], color,
                                                         category_names, font_size=15)

        figure(figsize=(10, 10))
        plt.imshow(annotated_text_image)
        plt.show()


if __name__ == '__main__':
    main()
