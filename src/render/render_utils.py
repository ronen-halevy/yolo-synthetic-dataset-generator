from pathlib import Path

import yaml
from PIL import Image
from PIL import ImageDraw
from PIL import Image as im
from PIL import ImageColor
from PIL import ImageFont
import numpy as np
import os
import cv2
import random

# from .utils import draw_bbox_xywh, draw_bbox_xyxy


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    """
    Description: Draws a text which starts at xmin,ymin bbox corner

    """
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



def draw_bbox_xywh(image, bboxes, category_names, thickness=1):
    # annotated_bbox_image = draw_bounding_box(image, bboxes)
    colors = list(ImageColor.colormap.values())
    color = colors[7]
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        xmin, ymin, w, h = bbox
        draw.line([(xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)


    text_box_color = [255, 255, 255]
    draw_text_on_bounding_box(image, np.array(bboxes)[..., 1],
                                                     np.array(bboxes)[..., 0], text_box_color,
                                                     category_names, font_size=15)

    return image


def draw_bbox_xyxy(image, bboxes, category_names, thickness=1):
    # annotated_bbox_image = draw_bbox_xyxy(image, bboxes)
    colors = list(ImageColor.colormap.values())
    color = colors[7]
    draw = ImageDraw.Draw(image)
    for bbox_xyxy in bboxes:
        draw.line([(bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (bbox_xyxy[4], bbox_xyxy[5]), (bbox_xyxy[6], bbox_xyxy[7]),
                   (bbox_xyxy[0], bbox_xyxy[1])],
                  width=thickness,
                  fill=color)




    text_box_color = [255, 255, 255]
    draw_text_on_bounding_box(image, np.array(bboxes)[..., 1],
                                                     np.array(bboxes)[..., 0], text_box_color,
                                                     category_names, font_size=15)

    return image


def read_detection_dataset_entry(image_path, label_path):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged as a text file per image arrangement.
    Label's file name corresponds to image filename with .txt extension. Label file format matches Ultralytics
    yolov5 detection dataset, i.e.  5 words per object rows, each holds category_id and bbox  x_center, y_center, w, h
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, x_center, y_center, w, h
    :type label_path: str
    :return:
    image: image read from file
    :type: PIL
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    """

    with open(label_path) as f:
        lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_ids = np.array(lables)[:, 0].astype(int)
    bboxes = np.array(lables, dtype=float)[:, 1:5] * [image.width, image.height, image.width, image.height]

    # convert from x_center, y_center to xmin, ymin
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    return image, bboxes, category_ids


def read_single_file_detection_dataset(label_path, image_dir):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged in a single text file, common to all dataset examples, a row per an image example. row's format:
     category_id, x_center, y_center, w, h. his method chooses randomly a file row, which holds an image path, and a
     set of bbox & category id per each object. Method returns the read image, a list of bboxes and ids.

    :param label_path: label file path.
    :type label_path: str
    :return:
    image , bboxes, category ids

    :rtype:

    """
    with open(label_path, 'r') as f:
        annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        example = random.choice(annotations).split()
        image_path = f'{image_dir}/{example[0]}'
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 0:4]
        category_ids = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 4].astype(int)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return image, bboxes, category_ids, image_path


def read_segmentation_dataset_entry(image_path, label_path):
    """
    Description:
    This method reads segmentation dataset entry. Following Ultralytics yolo convention,a dataset entry is defined
    by an image file and a label file with same name but latter extention is .txt. Label files formatted with a row per
    object, with a category_id and polygon's coordinates within.
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, polygon's coordinates
    :type label_path: str
    :return:
    image: image read from file. type: PIL
    polygons: list[nt], of nt polygons format: [[[xj,j],j=0:nv_i], i=0:nt], where nv_i nof vertices, nt: nof objects
    bboxes: list[nt]. entry format:  [[xmin,ymin,w,h], i=0:nt]
    category_ids:  a list of per-image-object category id
    """
    image = Image.open(image_path)
    size = np.array([image.height, image.width]).reshape(1, -1)
    with open(label_path) as f:
        entries = [x.split() for x in f.read().strip().splitlines() if len(x)]
    if 'entries' in locals():
        polygons = [np.array(entry)[1:].reshape(-1, 2).astype(float) for entry in entries]
        category_ids = [np.array(entry)[0].astype(int) for entry in entries]
        polygons = [(polygon * size).astype(int) for polygon in polygons]
        bboxes = []
        for polygon, category_id in zip(polygons, category_ids):
            x, y = polygon[:, 0], polygon[:, 1]
            bbox = [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()]
            bboxes.append(bbox)
    else:
        print(f'labels files {label_path} does not exist!')
        bboxes = []
        category_ids = []
        polygons = []

    return image, polygons, bboxes, category_ids


def read_obb_dataset_entry(image_path, label_path):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged as a text file per image arrangement.
    Label's file name corresponds to image filename with .txt extension. Label file format matches Ultralytics
    yolov5 detection dataset, i.e.  5 words per object rows, each holds category_id and bbox  x_center, y_center, w, h
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, x_center, y_center, w, h
    :type label_path: str
    :return:
    image: image read from file
    :type: PIL
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    """

    with open(label_path) as f:
        lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_names = np.array(lables)[:, 8]  #
    polygons = np.array(lables)[:, 0:8].astype(np.float32) * [image.width, image.height, image.width, image.height,
                                                              image.width, image.height, image.width, image.height]
    return image, polygons, category_names


def draw_detection_dataset_example(image_path, label_path, category_names_table, output_dir):
    [image, bboxes, category_ids] = read_detection_dataset_entry(image_path, label_path)
    category_names = [category_names_table[category_id] for category_id in category_ids]

    dest_dir = f'{output_dir}'
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(image_path)
    output_path = f'{dest_dir}/{fname.stem}_annotated{fname.suffix}'
    draw_bbox_xywh(image, bboxes, category_names)
    print(f'saving test results to {output_path}')
    image.save(output_path)


def draw_obb_dataset_example(image_path, label_path, category_names_table, output_dir):
    [image, polygons, category_names] = read_obb_dataset_entry(image_path, label_path)
    # category_names = [category_names_table[category_id] for category_id in category_ids]

    dest_dir = f'{output_dir}'
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(image_path)
    output_path = f'{dest_dir}/{fname.stem}_annotated{fname.suffix}'
    draw_bbox_xyxy(image, polygons, category_names)
    print(f'saving test results to {output_path}')
    image.save(output_path)


def draw_detection_single_file_dataset_example(label_path, image_dir, category_names_table, output_dir):
    [image, bboxes, category_ids, image_path] = read_single_file_detection_dataset(label_path, image_dir)
    category_names = [category_names_table[category_id] for category_id in category_ids]
    dest_dir = f'{output_dir}'
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(image_path)
    output_path = f'{dest_dir}/{fname.stem}_annotated{fname.suffix}'
    draw_bbox_xywh(image, bboxes, category_names)
    print(f'saving test results to {output_path}')
    image.save(output_path)


def draw_segmentation_dataset_example(image_path, label_path, category_names_table, output_dir):
    """
    Draw a randomly selected image with segmentation, bbox and class labels overlays

    :param image_dir: images directory for a random image selection
    :type image_dir: str
    :param label_dir: segmentation labels directory, a label file per an image, with same filename but .txt ext
    :type label_dir: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
    :param output_dir:
    :type output_dir:
    :return:
    :rtype:
    """

    # arrange output elements:
    [image, polygons, bboxes, category_ids] = read_segmentation_dataset_entry(image_path, label_path)
    # fill objects with  masks by polygons:
    array_image = np.array(image)
    if polygons:
        for polygon, category_id in zip(polygons, category_ids):
            color = np.random.randint(low=0, high=255, size=3).tolist()
            cv2.fillPoly(array_image, np.expand_dims(polygon, 0), color=color)
        image = im.fromarray(array_image)
        ImageDraw.Draw(image)

        # extract category names by ids:
        category_names = [category_names_table[category_id] for category_id in category_ids]
        # construct outpath:
        fname = Path(image_path)
        dest_dir = f'{output_dir}'
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        output_path = f'{dest_dir}/{fname.stem}_annotated{fname.suffix}'

        draw_bbox_xywh(image, bboxes, category_names)
        print(f'saving test results to {output_path}')
        image.save(output_path)


def draw_coco_detection_dataset_example(annotations_path, category_names_table, output_dir):
    """
    Draw a randomly selected image with bboxes and class labels overlays according to COCO format label files

    :param annotations_path: coco format annotations json file path
    :type annotations_path: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
    :param output_dir: dest dir for output image
    :type output_dir: str
    :return:
    :rtype:
    """
    with open(annotations_path) as file:
        annotations = yaml.safe_load(file)

    # randomy select an image index from dataset:
    if len(annotations['images']):
        image_index = np.random.randint(0, len(annotations['images']))
        # take records by index
        image_record = annotations['images'][image_index]
        annotation_records = [annotation for annotation in annotations['annotations'] if
                              annotation['image_id'] == image_record['id']]
        image_path = f'{image_record["file_name"]}'
        image = Image.open(image_path)

        bboxes = [annotation_record['bbox'] for annotation_record in annotation_records]
        bboxes = np.array(bboxes)
        category_ids = [annotation_record['category_id'] for annotation_record in annotation_records]

        # draw:
        category_names = [category_names_table[category_id] for category_id in category_ids]
        fname = Path(image_path)
        dest_dir = f'{output_dir}'
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        output_path = f'{dest_dir}/{fname.stem}_annotated{fname.suffix}'

        draw_bbox_xywh(image, bboxes, category_names)
        print(f'saving test results to {output_path}')
        image.save(output_path)
