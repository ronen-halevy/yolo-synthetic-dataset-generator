import numpy as np
from PIL import Image, ImageDraw
import math
from math import cos, sin
import random
import yaml
import os
from PIL import Image, ImageColor
import sys


class ShapesDataset:
    """
    A class to generate shapes dataset, based on a config file:

    Public method: create_dataset

    """

    def __init__(self, shapes_config_file):
        with open(shapes_config_file, 'r') as stream:
            shapes_config = yaml.safe_load(stream)
        # load shape yaml files.

        self.image_size = shapes_config['image_size']
        self.min_objects_in_image = shapes_config['min_objects_in_image']
        self.max_objects_in_image = shapes_config['max_objects_in_image']
        self.bg_color = shapes_config['bg_color']
        self.iou_thresh = shapes_config['iou_thresh']
        self.margin_from_edge = shapes_config['margin_from_edge']
        self.bbox_margin = shapes_config['bbox_margin']
        self.size_fluctuation = shapes_config['size_fluctuation']
        self.rotate_shapes = shapes_config['rotate_shapes'] # rotated shape may be image boundaries exeeded

        # create a class names output file.
        self.shapes = []
        for shape in shapes_config['shapes_categories']:
            self.shapes.append(shape)

        self.category_names = [shape['cname'] for shape in self.shapes]
        # reduce duplicated category names (config list may have dup rows for same category):
        indexes = np.unique(self.category_names, return_index=True)[1]
        self.category_names = [self.category_names[index] for index in sorted(indexes)]
        self.category_ids = [self.category_names.index(category_name) for category_name in self.category_names ]


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


    def __polygon_to_box(self,  polygon):
        """
        Description: Creates a bbox given a polygon

        :param polygon:   a list of nvertices 2 tuples which hold polygon's 2d vertices
        :type polygon: float.
        :return: bbox: [xmin, ymin, xmax, ymax]
        :rtype: float
        """
        x,y = np.array(polygon)[:,0], np.array(polygon)[:,1]
        return  [x.min(), y.min(), x.max(), y.max()]


    def rotate(self, polygon):
        # patch - dirty image for circle and ellipse if rotated by pi/4 - TBD
        rot_tick = math.pi/4 if len(polygon) < 10 else  math.pi/2
        # random rotation angle:
        rot_angle = rot_tick*np.random.randint(0, 8)

        rotate_x = lambda x, y: x * cos(rot_angle) + y * sin(rot_angle)
        rotate_y = lambda x, y: -x * sin(rot_angle) + y * cos(rot_angle)
        x, y = np.split(np.array(polygon), 2, axis=-1)

        x, y = rotate_x(x, y), rotate_y(x, y)
        polygon = np.concatenate([x, y], axis=-1)


        return polygon

    def __create_polygon(self, nvertices, theta0, height, aspect_ratio, size_fluctuation, margin_from_edge, image_size):

        """
        Description: Creates a polygon given nvertices, and data on shape's dims
        :param nvertices: type: string name of a supported shape
        :param height: A list of widths choices for random selection.
        :type height: floats list
        :param aspect_ratio: ratio between shapes height and width
        :type aspect_ratio:
        :param size_fluctuation: fluctuations of new bbox dims, each multiplied by (1-rand(0, size_fluctuation)) where
        <=0size_fluctuation<1
        :type size_fluctuation:
        :param margin_from_edge: Minimal distance in pixels between bbox and image edge.
        :type margin_from_edge: int
        :param image_size: Canvas size of target image
        :type image_size: 2 tuple, ints

        :return:
        polygon: type:float. a nvertices size list of tuple entries. tuples hold vertices x,y coords
        """

        sel_height = np.random.choice(height)
        sel_aspect_ratio = np.random.choice(aspect_ratio)
        shape_width = sel_height * sel_aspect_ratio * random.uniform(1 - size_fluctuation, 1)
        sel_height = sel_height * random.uniform(1 - size_fluctuation, 1)

        radius = np.array([shape_width  / 2, sel_height / 2])
        center = np.random.randint(
            low=radius + margin_from_edge, high=np.floor(image_size - radius - margin_from_edge), size=2)

        polygon = [
            (cos(th) * radius[0],
             sin(th
                      ) * radius[1])
            for th in [i * (2 * math.pi) / nvertices + math.radians(theta0)for i in range(nvertices)]
        ]
        # rotate shape:
        if self.rotate_shapes:
            polygon= self.rotate(polygon)

        # translate to center:
        polygon+=center
        polygon=tuple(map(tuple, polygon))
        return polygon

    def __create_ds_entry(self, objects_attributes, image_size, bg_color, iou_thresh,
                          margin_from_edge,
                          bbox_margin,
                          size_fluctuation
                          ):
        """
        Create a single dataset entry, with nt bboxes and polygons.

        :param objects_attributes: list[nt], entry: [cls_id, nvertices, theta0, cls_name, [aspect_ratio], [height],
        [color]] where aspect ratio, height and color(str) are lists for random selection,
        :param image_size: image size, list[2] height and width
        :param bg_color: type: str image's bg color
        :param iou_thresh: type: float [0,1], maximal iou value for adjacent bboxes. iou=1 means complete overlap. iou=0 means no overlap
        :param margin_from_edge: type: int. minimal distance in pixels of bbox from image's edge.
        :param bbox_margin: type: int. distance in pixels between bbox and shape
        :param size_fluctuation: int float [0,1), images' width and height are multiplied by (1-rand(size_fluctuation))
        :return:
            image: an RGB pillow image drawn with shapes
            bboxes: type: float ndarray [nobjects,4]
            tuple(objects_categories_indices): type in. tuple of nobjects category indices
            objects_categories_names: type: str. list of nobjects category names
            polygons: type: float. list of nobjects, each object shape with own 2 points nvertices
        """
        image = Image.new('RGB', image_size, bg_color)
        draw = ImageDraw.Draw(image)
        bboxes = []
        polygons=[]
        objects_categories_names = []
        objects_categories_indices = []

        for category_id, nvertices, theta0, category_name, aspect_ratio, height, color in objects_attributes:
            max_count = 10
            count = 0
            # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
            while True:
                polygon = self.__create_polygon(nvertices,theta0, height, aspect_ratio, size_fluctuation,
                                                margin_from_edge,
                                                image_size)
                bbox = self.__polygon_to_box(polygon)

                iou = list(map(lambda x: self.__compute_iou(bbox, x), bboxes))

                if len(iou) == 0 or max(iou) <= iou_thresh:
                    break
                if count > max_count:
                    print(
                        f'Shape Objects Placement Failed after {count} placement itterations: max(iou)={max(iou)}, '
                        f'but required iou_thresh is {iou_thresh} height: . \nHint: reduce objects size or'
                        f' quantity of objects in an image')
                    exit(1)
                count+=1

            if len(bbox):
                bboxes.append(bbox)

            # draw shape on image:
            sel_color = np.random.choice(color)
            draw.polygon(polygon, fill=ImageColor.getrgb(sel_color) )

            polygons.append(polygon)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(category_id)

        bboxes = np.array(bboxes)
        # [x,y,x,y] to [xc, yc, w, h] (bbox_margin is added distance between shape and bbox):
        bboxes = [(bboxes[:, 0] +  bboxes[:, 2])/2,
                  (bboxes[:, 1] +  bboxes[:, 3])/2,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return image, bboxes, tuple(objects_categories_indices), objects_categories_names, polygons


    def create_dataset(self,  nentries, output_dir):
        """
        Description: Create dataset entries. svae created images in output_dir, and return dataset metadata.

        :param nentries: type: int, number of entries to produce
        :param output_dir: type: str,  destination output dir for dataset's images
        :return:
        images_filenames: type: list of str size:  nentries. created images filenames, w/o dir prefix
        images_sizes: type:  list of 2 tuples ints.  size:  nentries. (image.height, image.width)
        images_bboxes: type:  list of float [nobjects, 4] arrays . size:  nentries. Bounding boxes of image's nobjects
        images_objects_categories_indices: type: list of nobjects tuples size: nentries. Category id of image's nobjects
        self.category_names: type: list of str. size: ncategories. Created dataset's num of categories.
        self.category_ids:  type: list of int. size: ncategories. Created dataset's entries ids.
        polygons: type: float. list of nobjects, each object shape with own 2 points nvertices. Needed for segmentation

        """

        images_filenames = []
        images_sizes = []
        images_bboxes = []
        images_objects_categories_indices = []
        images_objects_categories_names = []
        images_polygons = []
        # loop to create nentries examples:
        for example_id in range(nentries):
            # randomize num of objects in an image:
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)
            # take only active shapes for dataset creation:
            active_shapes = [shape   for shape in self.shapes if shape['active']]
            bg_color = np.random.choice(self.bg_color)
            # randomly select num_of_objects shapes:
            sel_shape_entris = [np.random.choice(active_shapes) for idx in range(num_of_objects)]

            sel_index = random.randint(0, len(self.image_size)-1) # randomly select img size index from config
            image_size=self.image_size[sel_index]
            # arrange target objects attributes from selected shapes:
            objects_attributes = [
                [self.category_names.index(shape_entry['cname']), shape_entry['nvertices'], shape_entry['theta0'], shape_entry['cname'], shape_entry['aspect_ratio'],
                 shape_entry['height'],
                 shape_entry['color']] for shape_entry in sel_shape_entris]
            image, bboxes, objects_categories_indices, objects_categories_names, polygons = self.__create_ds_entry(
                    objects_attributes,
                    image_size,
                    bg_color,
                    self.iou_thresh,
                    self.margin_from_edge,
                    self.bbox_margin,
                    self.size_fluctuation)

            # save image files
            image_filename = f'img_{example_id:06d}.jpg'
            file_path = f'{output_dir}/{image_filename}'
            print(f'writing image file to disk: {image_filename}')
            image.save(file_path)
            if len(bboxes) == 0:
                continue

            images_filenames.append(image_filename)
            images_sizes.append((image.width, image.height))
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)
            images_polygons.append(polygons)

        return images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, self.category_names, self.category_ids, images_polygons
