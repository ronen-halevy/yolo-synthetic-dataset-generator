from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import numpy as np

from src.render.render_utils import draw_text_on_bounding_box

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

def draw_obb_dataset_example(image_path, label_path):
    [image, polygons, category_names] = read_obb_dataset_entry(image_path, label_path)
    draw_bbox_xyxy(image, polygons, category_names)
    return image

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
    polygons = np.array(lables)[:, 0:8].astype(np.float32)#
    # Dota Obb labels format are normally not normalized - if so, don't scale up
    if np.all(polygons<1):
        polygons = polygons.reshape([-1,4,2]) * np.array(image.size)
    return image, polygons, category_names
