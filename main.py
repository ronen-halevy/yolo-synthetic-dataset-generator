import numpy as np
from PIL import Image, ImageDraw
import os
import math
import json

def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""
    area_box_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax:
        return 0
    return ((xmax - xmin) * (ymax - ymin)) / (area_box_2 + area_box_1)


def set_bbox(x_diameter_choices, axis_ratio, image_size, border_margin):
    x_diameter = np.random.choice(x_diameter_choices)
    y_diameter = x_diameter * axis_ratio
    radius = np.array([x_diameter / 2, y_diameter / 2])
    center = np.random.randint(
        radius + border_margin, [np.floor(image_size - radius - border_margin)], 2)

    new_bbox = np.concatenate(np.tile(center, 2).reshape(2, 2) +
                              np.array([np.negative(radius), radius]))
    return new_bbox


def create_bbox(image_size, bboxes, x_diameter_choices, axis_ratio, iou_thresh, margin_from_edge):
    max_count = 10000
    count = 0
    while True:
        new_bbox = set_bbox(x_diameter_choices, axis_ratio, image_size, margin_from_edge)

        iou = [compute_iou(new_bbox, bbox) for bbox in bboxes]
        if len(iou) == 0 or max(iou) == iou_thresh:
            break
        if count > max_count:
            new_bbox = []
        count += 1

    return new_bbox

def make_image(shapes, image_size, max_objects_in_image, bg_color, iou_thresh, margin_from_edge):
    image = Image.new('RGB', image_size, tuple(bg_color))
    draw = ImageDraw.Draw(image)
    num_of_objects = np.random.randint(6, max_objects_in_image)
    bboxes = []
    added_shapes = []
    for idx in range(num_of_objects):
        shape_entry = np.random.choice(shapes)
        added_shapes.append(shape_entry)

    for shape_entry in added_shapes:
        axis_ratio = shape_entry['diameter_ratio']
        x_diameter_choices = shape_entry['x_diameter_choices']
        bbox = create_bbox(image_size, bboxes, x_diameter_choices, axis_ratio, iou_thresh, margin_from_edge)
        if len(bbox):
            bboxes.append(bbox)
        else:
            break
        color = tuple(shape_entry['color'])

        if shape_entry['shape_type'] == 'ellipse':
            draw.ellipse(bbox.tolist(), fill=color, outline=color)

        elif shape_entry['shape_type'] == 'rectangle':
            draw.rectangle(bbox.tolist(), fill=color, outline=color)

        elif shape_entry['shape_type'] == 'triangle':
            xmin, ymin, xmax, ymax = bbox.tolist()
            vertices = [xmin, ymax, xmax, ymax, (xmin + xmax) / 2, ymin]
            points = [bbox.tolist()[0], bbox.tolist()[1]]
            draw.polygon(vertices, fill=color, outline=color)

        elif shape_entry['shape_type'] == 'triangle':
            xmin, ymin, xmax, ymax = bbox.tolist()
            vertices = [xmin, ymax, xmax, ymax, (xmin + xmax) / 2, ymin]
            points = [bbox.tolist()[0], bbox.tolist()[1]]
            draw.polygon(vertices, fill=color, outline=color)

        elif shape_entry['shape_type'] in ['trapezoid', 'hexagon']:
            xmin, ymin, xmax, ymax = bbox.tolist()
            sides = shape_entry['sides']
            center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            rad_x, rad_y = (xmax - xmin) / 2, (ymax - ymin) / 2
            xy = [
                (math.cos(th) * rad_x + center_x,
                 math.sin(th) * rad_y + center_y)
                for th in [i * (2 * math.pi) / sides for i in range(sides)]
            ]
            draw.polygon(xy, fill=color, outline=color)
    return image, bboxes, added_shapes


def create_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
        print("The new directory is created!")


def main(config_file, shapes_file):
    with open(config_file) as f:
        config = json.load(f)
    sections = config['sections']

    with open(shapes_file) as f:
        shapes = json.load(f)['shapes']

    for section in sections:

        num_of_examples = sections[section]["num_of_examples"]

        images_dir = sections[section]["images_dir"]
        annotations_path = sections[section]["annotations_path"]

        with open(annotations_path, 'w') as f:
            for example in range(int(num_of_examples)):

                image, bboxes, added_shapes = make_image(shapes, config['image_size'],
                                                         config['max_objects_in_image'],
                                                         config['bg_color'], config['iou_thresh'],
                                                         config['margin_from_edge'])
                if len(bboxes) == 0:
                    continue
                file_path = f'{images_dir}/{example + 1:06d}.jpg'
                image.save(file_path)
                annotation = file_path
                for box, shape in zip(bboxes, added_shapes):
                    box_and_label = np.append(box, shape['id'])
                    annotation += ' ' + ','.join([str(entry) for entry in box_and_label.astype(np.int32)])
                f.write(annotation + '\n')

config_file = 'config.json'
shapes_file = 'shapes.json'

main(config_file, shapes_file)

