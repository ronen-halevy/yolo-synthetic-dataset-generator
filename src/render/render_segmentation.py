from PIL import Image

from PIL import ImageDraw
import numpy as np
import cv2
from src.render.render_utils import draw_bbox_xywh

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


def draw_segmentation_dataset_example(image_path, label_path, category_names_table):
    """
    Draw a randomly selected image with segmentation, bbox and class labels overlays

    :param image_dir: images directory for a random image selection
    :type image_dir: str
    :param label_dir: segmentation labels directory, a label file per an image, with same filename but .txt ext
    :type label_dir: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
    :return:
    :rtype:
    """

    # arrange output elements:
    [image, polygons, bboxes, category_ids] = read_segmentation_dataset_entry(image_path, label_path)
    # fill objects with  masks by polygons:
    array_image = np.array(image)
    # if polygons:
    for polygon, category_id in zip(polygons, category_ids):
        color = np.random.randint(low=0, high=255, size=3).tolist()
        cv2.fillPoly(array_image, np.expand_dims(polygon, 0), color=color)
    image = Image.fromarray(array_image)
    ImageDraw.Draw(image)
    # extract category names by ids:
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image

