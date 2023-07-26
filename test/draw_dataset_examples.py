from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import Image as im
import numpy as np
import yaml
import os
import cv2
import random

from utils import draw_dataset_entry


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
    if os.path.isfile(label_path):
        with open(label_path) as f:
            lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_ids = np.array(lables)[:, 0].astype(int)
    bboxes = np.array(lables, dtype=float)[:, 1:5] * [image.width, image.height, image.width, image.height]

    # convert from x_center, y_center to xmin, ymin
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    return image, bboxes, category_ids


def read_single_file_detection_dataset(label_path):
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
        image_path = example[0]
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 0:4]
        category_ids = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 4].astype(int)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return image, bboxes, category_ids, image_path


def read_segmentation_dataset_entry(image_path, label_path):
    """
    Description:
    This method demonstrates the reading and rendering of a segmentation dataset entry, where the dataset labels are
    arranged as a text file per image arrangement.
    Label's file name corresponds to image filename with .txt extension. Label file format matches Ultralytics
    yolov5 detection dataset, i.e. per object rows, each holds ategory_id and polygon's coordinates
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, polygon's coordinates
    :type label_path: str
    :return:
    image: image read from file
    :type: PIL
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    """

    if os.path.isfile(label_path):
        with open(label_path) as f:
            entries = [x.split() for x in f.read().strip().splitlines() if len(x)]

    polygons = [np.array(entry)[1:].reshape(-1, 2).astype(float) for entry in entries]
    category_ids = [np.array(entry)[0].astype(int) for entry in entries]

    image = Image.open(image_path)
    array_image = np.array(image)
    size = np.array([image.height, image.width]).reshape(1, -1)
    polygons = [(polygon * size).astype(int) for polygon in polygons]
    bboxes = []
    for polygon, category_id in zip(polygons, category_ids):
        color = np.random.randint(low=0, high=255, size=3).tolist()
        cv2.fillPoly(array_image, np.expand_dims(polygon, 0), color=color)
        x, y = polygon[:, 0], polygon[:, 1]
        bbox = [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()]
        bboxes.append(bbox)

    image = im.fromarray(array_image)
    ImageDraw.Draw(image)
    return image, bboxes, category_ids


def main():
    config_file = 'test/test_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config['output_dir']
    category_names_file = config['category_names_file']
    with open(category_names_file) as f:
        category_names_table = f.readlines()

    if 'yolov5_detection_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['yolov5_detection_format']:
            [image, bboxes, category_ids] = read_detection_dataset_entry(params['image_path'], params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'yolov5_detection_format {params["image_path"]}'

            dest_dir = f'{output_dir}/det1'
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            fname = Path(params['image_path'])
            output_path = f'{dest_dir}/{fname.stem}"_annotated"{fname.suffix}'
            draw_dataset_entry(image, bboxes, category_names, title, output_path)

    if 'single_label_file_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['single_label_file_format']:
            [image, bboxes, category_ids, image_path] = read_single_file_detection_dataset(params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'single_label_file_format {params["label_path"]}'
            dest_dir = f'{output_dir}/det2'
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            fname = Path(image_path)
            output_path = f'{dest_dir}/{fname.stem}"_annotated"{fname.suffix}'
            draw_dataset_entry(image, bboxes, category_names, title, output_path)

    if 'yolov5_segmentation_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['yolov5_segmentation_format']:
            [image, bboxes, category_ids] = read_segmentation_dataset_entry(params['image_path'], params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'yolov5_segmentation_format {params["image_path"]}'
            fname = Path(params['image_path'])
            dest_dir = f'{output_dir}/seg'
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            output_path = f'{dest_dir}/{fname.stem}"_annotated"{fname.suffix}'

            draw_dataset_entry(image, bboxes, category_names, title, output_path)


if __name__ == "__main__":
    main()
