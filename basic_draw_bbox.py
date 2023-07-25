from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image as im

import os
import cv2


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
    text_heights = tuple(map(lambda i, j: i - j, bottom, top))
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
                       fill=tuple(color))

        draw.text((xmint + text_margin, text_bottom - text_height - 3 * text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


def draw_bounding_box(image, boxes, thickness=1):
    colors = list(ImageColor.colormap.values())
    color = colors[0]
    thickness = 1
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, w, h = box
        # xmin = xc-w/2
        # ymin = yc-h/2
        print((xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin))
        draw.line([(xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image


category_names_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/class.names'
with open(category_names_file) as f:
    category_names_table = f.readlines()

label_file_format = 'yolov5_detection_format' # ['yolov5_detection_format', 'single_label_file_format', 'yolov5_segmentation_format']

if label_file_format == 'yolov5_detection_format':
    image_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/images/img_000001.jpg'
    lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/labels/img_000001.txt'
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file) as f:
            lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_ids =np.array(lables)[:, 0].astype(int)
    bboxes = np.array(lables, dtype=float)[:, 1:5] * [image.width, image.height, image.width, image.height]

    # convert to x_Center, y_center to cmin, ymin
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
elif label_file_format == 'single_label_file_format':
    lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/all_entries.txt'
    with open(lb_file, 'r') as f:
        annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        example = annotations[0].split()
        image_path = example[0]
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 0:4]
        category_ids = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 4].astype(int)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
elif label_file_format =='yolov5_segmentation_format':
    image_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/images/img_000001.jpg'
    lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/labels-seg/img_000001.txt'
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file) as f:
            polygons = [x.split() for x in f.read().strip().splitlines() if len(x)]
    category_ids = np.asarray(polygons)[:, 0].astype(int)
    polygons = np.asarray(polygons)[:, 1:].astype(float)

    polygons = polygons.reshape(polygons.shape[0], -1, 2)

    image = Image.open(image_path)
    array_image = np.array(image)

    size = np.array([image.height, image.width]).reshape(1, 1, -1)
    polygons = (polygons * size).astype(int)
    bboxes = []
    for polygon, category_id in zip(polygons, category_ids):
        color = np.random.randint(low=0, high=255, size=3).tolist()
        cv2.fillPoly(array_image, np.expand_dims(polygon, 0), color=color)
        x, y = polygon[:, 0], polygon[:, 1]
        bbox = [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()]
        bboxes.append(bbox)

    image = im.fromarray(array_image)
    draw = ImageDraw.Draw(image)

annotated_bbox_image = draw_bounding_box(image, bboxes)
category_names = [category_names_table[category_id] for category_id in category_ids]

text_box_color=[255,255,255]
annotated_text_image = draw_text_on_bounding_box(annotated_bbox_image, np.array(bboxes)[..., 1],
                                                 np.array(bboxes)[..., 0], text_box_color,
                                                 category_names, font_size=15)

figure(figsize=(10, 10))
plt.imshow(image)
plt.show()
