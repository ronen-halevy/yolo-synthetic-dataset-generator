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

import json

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

    text_widths, text_heights = zip(*[font.getsize(display_str) for display_str in display_str_list])
    text_margins = np.ceil(text_margin_factor * np.array(text_heights))
    text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

    for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
            zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):
        text_width, text_height = font.getsize(display_str)
        text_margin = np.ceil(text_margin_factor * text_height)

        draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                        (xmint + text_width + text_margin, text_bottom)),
                       fill=color)

        draw.text((xmint + text_margin, text_bottom - text_height - text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


def draw_bounding_box(image, boxes, color, thickness=1):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image


def main():
    with open('config/config.json') as f:
        config = json.load(f)

    annotations_path = config["annotations_path"]
    images_dir = config['images_dir']

    with open(annotations_path) as file:
        annotations = json.load(file)['annotations']

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
        line = annotations[image_index]
        boxes = line['bboxes']

        image_path = f'{images_dir}{line["image_filename"]}'

        image = Image.open(image_path)
        colors = list(ImageColor.colormap.values())
        color = colors[0]

        boxes = np.array(boxes)
        width, height = image.size
        boxes[..., 0:3:2] = boxes[..., 0:3:2] * width
        boxes[..., 1:4:2] = boxes[..., 1:4:2] * height
        annotated_bbox_image = draw_bounding_box(image, boxes, color, thickness=1)

        classes = [anno_object['label'] for anno_object in line['objects']]

        annotated_text_image = draw_text_on_bounding_box(annotated_bbox_image, boxes[..., 1], boxes[..., 0], color,
                                                         classes, font_size=15)

        figure(figsize=(10, 10))
        plt.imshow(annotated_text_image)
        plt.show()


if __name__ == '__main__':
    main()
