import numpy as np
from PIL import Image, ImageDraw
import math
import random
import yaml
import cv2

import matplotlib.pyplot as plt


shapes_file = '../config/shapes.yaml'


class ShapesDataset:
    """
    A class to generate shapes dataset, based on config
    Public method: create_dataset

    """
    def __init__(self):
        shapes_file = 'config/shapes.yaml'
        with open(shapes_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.image_size = config['image_size']
        self.min_objects_in_image = config['min_objects_in_image']
        self.max_objects_in_image = config['max_objects_in_image']
        self.bg_color = tuple(config['bg_color'])
        self.iou_thresh = config['iou_thresh']
        self.margin_from_edge = config['margin_from_edge']
        self.bbox_margin = config['bbox_margin']
        self.size_fluctuation = config['size_fluctuation']
        self.shapes=config['shapes']


        self.category_names = [shape['category_name'] for shape in self.shapes]
        with open(config['class_names_file'], 'w') as f:
            for category_name in self.category_names:
                f.write(f'{category_name}\n')

        self.super_category_names = [shape['super_category'] for shape in self.shapes]
        pass


    def __compute_iou(self, box1, box2):
        """
        Computes iou (intersection over union) of 2 boxes: iou=0 if no overlap and 1 if totally overlap.
        Used for  objects placement in image
        :param box1: format x_min, y_min, x_max, y_max
        :param box2: x_min, y_min, x_max, y_max
        :return: (boxes intesection area)/(boxes union area)
        """
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

    import math

    # This function gets just one pair of coordinates based on the angle theta

    def __create_bbox(self, image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                      size_fluctuation=0.01):
        """

        :param image_size: Canvas size
        :type image_size:
        :param bboxes:
        :type bboxes:
        :param shape_width_choices:
        :type shape_width_choices:
        :param axis_ratio:
        :type axis_ratio:
        :param iou_thresh:
        :type iou_thresh:
        :param margin_from_edge:
        :type margin_from_edge:
        :param size_fluctuation:
        :type size_fluctuation:
        :return:
        :rtype:
        """
        max_count = 10000
        count = 0
        # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
        while True:
            shape_width = np.random.choice(shape_width_choices)
            shape_height = shape_width * axis_ratio * random.uniform(1 - size_fluctuation, 1)
            # add fluctuations - config defuned
            shape_width = shape_width * random.uniform(1 - size_fluctuation, 1)
            radius = np.array([shape_width / 2, shape_height / 2])
            center = np.random.randint(
                low=radius + margin_from_edge, high=np.floor(image_size - radius - margin_from_edge), size=2)
            # bbox_sides = radius
            new_bbox = np.concatenate(np.tile(center, 2).reshape(2, 2) +
                                      np.array([np.negative(radius), radius]))
            # iou new shape bbox with all prev bboxes. skip shape if max iou > thresh - try another placement for shpe
            iou = list(map(lambda x: self.__compute_iou(new_bbox, x), bboxes))

            if len(iou) == 0 or max(iou) <= iou_thresh:
                break
            if count > max_count:
                max_iou = max(iou)
                raise Exception(
                    f'Shape Objects Placement Failed after {count} placement itterations: max(iou)={max_iou}, '
                    f'but required iou_thresh is {iou_thresh} shape_width: {shape_width},'
                    f' shape_height: {shape_height}. . \nHint: reduce objects size or quantity of objects in an image')
            count += 1

        return new_bbox

    def __draw_shape(self, draw, shape, bbox, fill_color, outline_color):
        if shape in ['ellipse', 'circle']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.ellipse([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)
            n_points=100
            t = np.linspace(0,360,n_points)
            x_coord =(x_min+x_max)/2+(x_max-x_min)/2*np.cos(np.radians(t))
            y_coord = (y_min+y_max)/2+(y_max-y_min)/2*np.sin(np.radians(t))
            polygon  = [[ x_poly,y_poly] for x_poly,y_poly in zip(x_coord,y_coord)]
            polygon=np.asarray(polygon).astype(np.int32)


        elif shape in ['rectangle', 'square']:
            x_min, y_min, x_max, y_max = bbox.tolist()
            draw.rectangle((x_min, y_min, x_max, y_max), fill=fill_color, outline=outline_color, width=3)
            polygon=[ [x_min, y_min],[x_min, y_max], [x_max, y_max], [x_max, y_min]]
            polygon=np.asarray(polygon).astype(np.int32)


        elif shape == 'triangle':
            x_min, y_min, x_max, y_max = bbox.tolist()
            vertices = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]
            draw.polygon(vertices, fill=fill_color, outline=outline_color)
            polygon=[ [x_min, y_max],[x_max, y_max], [(x_min + x_max) / 2, y_min]]
## use same polygon for all:

        elif shape in ['trapezoid', 'hexagon']:
            sides = 5 if shape == 'trapezoid' else 6
            x_min, y_min, x_max, y_max = bbox.tolist()
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            rad_x, rad_y = (x_max - x_min) / 2, (y_max - y_min) / 2
            polygon= [
                (math.cos(th) * rad_x + center_x,
                 math.sin(th) * rad_y + center_y)
                for th in [i * (2 * math.pi) / sides for i in range(sides)]
            ]
            draw.polygon(polygon, fill=fill_color, outline=outline_color)



        return  np.asarray(polygon).astype(np.int32)


    def __create_ds_example(self, shapes_attributes, image_size, num_of_objects, bg_color, iou_thresh,
                          margin_from_edge,
                          bbox_margin,
                          size_fluctuation

                          ):
        image = Image.new('RGB', image_size, bg_color)
        draw = ImageDraw.Draw(image)
        bboxes = []
        objects_categories_names = []
        objects_categories_indices = []

        for entry_id, category_name, shape_aspect_ratio, shape_width_choices, fill_color, outline_color in shapes_attributes:
            try:
                bbox = self.__create_bbox(image_size, bboxes, shape_width_choices, shape_aspect_ratio, iou_thresh,
                                          margin_from_edge,
                                          size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(f'Failed in __create_bbox :\n{msg}.\nHere is the failed-to-be-placed shape entry: {entry_id}, {category_name}')

            if len(bbox):
                bboxes.append(bbox.tolist())
            else:
                break
            self.__draw_shape(draw, category_name, bbox, fill_color, outline_color)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(entry_id)

        bboxes = np.array(bboxes)
        # transfer bbox coordinate to:  [xmin, ymin, w, h]: (bbox_margin is added distance between shape and bbox)
        bboxes = [bboxes[:, 0] - bbox_margin,
                  bboxes[:, 1] - bbox_margin,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]  # / np.tile(image_size,2)

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return image, bboxes, objects_categories_indices, objects_categories_names

    def create_dataset(self,  nentries, output_dir):
        """

        :param shapes:
        :param image_size:
        :param min_objects_in_image:
        :param max_objects_in_image:
        :param bg_color:
        :param iou_thresh:
        :param margin_from_edge:
        :param bbox_margin:
        :param size_fluctuation:
        :param nentries:
        :param output_dir:
        :return:
        """

        images_filenames = []
        images_sizes = []
        images_bboxes = []
        images_objects_categories_indices = []
        images_objects_categories_names = []
        for example_id in range(nentries):
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)

            shape_entris= [np.random.choice(self.shapes) for idx in range(num_of_objects)]
            shapes_attributes = [[shape_entry['id'],  shape_entry['category_name'], shape_entry['shape_aspect_ratio'], shape_entry['shape_width_choices'],
                                 tuple(shape_entry['fill_color']), tuple(shape_entry['outline_color'])] for shape_entry in shape_entris]
            try:
                image, bboxes, objects_categories_indices, objects_categories_names = self.__create_ds_example(shapes_attributes,
                                                                                                             self.image_size,
                                                                                                             num_of_objects,
                                                                                                             self.bg_color,
                                                                                                             self.iou_thresh,
                                                                                                             self.margin_from_edge,
                                                                                                             self.bbox_margin,
                                                                                                             self.size_fluctuation)
            except Exception as e:
                msg = str(e)
                raise Exception(f'Error: While creating the {example_id}th image: {msg}')
            image_filename = f'img_{example_id + 1:06d}.jpg'
            file_path = f'{output_dir}/images/{image_filename}'
            image.save(file_path)
            if len(bboxes) == 0:
                continue

            images_filenames.append(image_filename)
            images_sizes.append([image.height, image.width])
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)

        return images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, self.category_names,  self.super_category_names