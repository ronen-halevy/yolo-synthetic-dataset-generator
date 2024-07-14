from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import numpy as np
import yaml
import random

from src.render.render_utils import draw_text_on_bounding_box



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


def draw_detection_dataset_example(image_path, label_path, category_names_table):
    [image, bboxes, category_ids] = read_detection_dataset_entry(image_path, label_path)
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image

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

def draw_detection_single_file_dataset_example(label_path, image_dir, category_names_table):
    [image, bboxes, category_ids, image_path] = read_single_file_detection_dataset(label_path, image_dir)
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image


def draw_coco_detection_dataset_example(annotations_path, category_names_table):
    """
    Draw a randomly selected image with bboxes and class labels overlays according to COCO format label files

    :param annotations_path: coco format annotations json file path
    :type annotations_path: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
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
        draw_bbox_xywh(image, bboxes, category_names)
        return image