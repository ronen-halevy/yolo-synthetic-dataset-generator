import numpy as np
from PIL import Image, ImageDraw
import math
import json


def compute_iou(box1, box2):
    """x_min, y_min, x_max, y_max"""
    area_box_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    if y_min >= y_max or x_min >= x_max:
        return 0
    return ((x_max - x_min) * (y_max - y_min)) / (area_box_2 + area_box_1)


def create_bbox(image_size, bboxes, x_diameter_choices, axis_ratio, iou_thresh, margin_from_edge):
    max_count = 10000
    count = 0
    while True:
        x_diameter = np.random.choice(x_diameter_choices)
        y_diameter = x_diameter * axis_ratio
        radius = np.array([x_diameter / 2, y_diameter / 2])
        center = np.random.randint(
            radius + margin_from_edge, [np.floor(image_size - radius - margin_from_edge)], 2)

        new_bbox = np.concatenate(np.tile(center, 2).reshape(2, 2) +
                                  np.array([np.negative(radius), radius]))

        iou = [compute_iou(new_bbox, bbox) for bbox in bboxes]
        if len(iou) == 0 or max(iou) == iou_thresh:
            break
        if count > max_count:
            new_bbox = []
            break
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
        fill_color = tuple(shape_entry['fill_color']) if len(shape_entry['fill_color']) else None
        outline_color = tuple(shape_entry['outline_color']) if len(shape_entry['outline_color']) else None

        if shape_entry['shape_type'] == 'ellipse':
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.ellipse([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)

        elif shape_entry['shape_type'] == 'rectangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)

        elif shape_entry['shape_type'] == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape_entry['shape_type'] == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)

        elif shape_entry['shape_type'] in ['trapezoid', 'hexagon']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            sides = shape_entry['sides']
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            rad_x, rad_y = (x_max - x_min) / 2, (y_max - y_min) / 2
            xy = [
                (math.cos(th) * rad_x + center_x,
                 math.sin(th) * rad_y + center_y)
                for th in [i * (2 * math.pi) / sides for i in range(sides)]
            ]
            draw.polygon(xy, fill=fill_color, outline=outline_color)
    return image, bboxes, added_shapes


def main(config, shapes):
    sections = config['sections']

    for section in sections:

        num_of_examples = sections[section]["num_of_examples"]

        images_dir = sections[section]["images_dir"]
        annotations_path = sections[section]["annotations_path"]

        with open(annotations_path, 'w') as annotation_file:
            for example in range(int(num_of_examples)):

                image, bboxes, added_shapes = make_image(shapes, config['image_size'],
                                                         config['max_objects_in_image'],
                                                         config['bg_color'],
                                                         config['iou_thresh'],
                                                         config['margin_from_edge'])
                if len(bboxes) == 0:
                    continue
                file_path = f'{images_dir}/{example + 1:06d}.jpg'
                image.save(file_path)
                annotation = file_path
                for box, shape in zip(bboxes, added_shapes):
                    box_and_label = np.append(box, shape['id'])
                    annotation += ' ' + ','.join([str(entry) for entry in box_and_label.astype(np.int32)])
                annotation_file.write(annotation + '\n')


if __name__ == '__main__':
    config_file = 'config.json'
    shapes_file = 'shapes.json'
    with open(config_file) as f:
        config_data = json.load(f)

    with open(shapes_file) as f:
        shapes_data = json.load(f)['shapes']

    main(config=config_data, shapes=shapes_data)
