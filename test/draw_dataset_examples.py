from PIL import Image
from PIL import ImageDraw
from PIL import Image as im
import numpy as np
import yaml
import os
import cv2

from utils import draw_dataset_entry


def read_yolov5_detection_dataset(image_path, label_path):
    """
    Description: Reads and image and label file. Label file format matches Ultralytics yolov5 detection dataset, i.e.:
    A dedicated label file per an image. Label's file name corresponds to image filename with .txr extension  .
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, x_center, y_center, w, h
    :type label_path: str
    :return:
    image: image read from file
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    :rtype:
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
    with open(label_path, 'r') as f:
        annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        example = annotations[0].split()
        image_path = example[0]
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 0:4]
        category_ids = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 4].astype(int)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return image, bboxes, category_ids


def read_yolov5_segmentation_dataset(image_path, label_path):
    if os.path.isfile(label_path):
        with open(label_path) as f:
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
    ImageDraw.Draw(image)
    return image, bboxes, category_ids


def main():
    config_file = 'test/test_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    label_file_formats = config['label_file_formats']
    category_names_file = config['category_names_file']
    with open(category_names_file) as f:
        category_names_table = f.readlines()

    if 'yolov5_detection_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['yolov5_detection_format']:
            [image, bboxes, category_ids] = read_yolov5_detection_dataset(params['image_path'], params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'yolov5_detection_format {params["image_path"]}'
            draw_dataset_entry(image, bboxes, category_names, title)

    if 'single_label_file_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['single_label_file_format']:
            [image, bboxes, category_ids] = read_single_file_detection_dataset(params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'single_label_file_format {params["label_path"]}'
            draw_dataset_entry(image, bboxes, category_names, title)

    if 'yolov5_segmentation_format' in config['label_file_formats'].keys():
        for params in config['label_file_formats']['yolov5_segmentation_format']:
            [image, bboxes, category_ids] = read_yolov5_segmentation_dataset(params['image_path'], params['label_path'])
            category_names = [category_names_table[category_id] for category_id in category_ids]
            title = f'yolov5_segmentation_format {params["image_path"]}'
            draw_dataset_entry(image, bboxes, category_names, title)


if __name__ == "__main__":
    main()
