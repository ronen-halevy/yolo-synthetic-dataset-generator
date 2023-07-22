import numpy as np
from PIL import Image, ImageDraw
import math
import random
import yaml
import os
from PIL import Image, ImageColor
import sys

shapes_config_file = 'config/shapes_config.yaml'
shapes_dir = 'config/shapes/'


class ShapesDataset:
    """
    A class to generate shapes dataset, based on a set of yaml config files:
    1. per shape configuration files located at shapes_config_file
    2. shapes_config.yaml - with defintions for image composing and a dataset_selector object which detemines
    the set of dataset shapes categories assigned to category id.
    Public method: create_dataset

    """

    def __init__(self):
        with open(shapes_config_file, 'r') as stream:
            shapes_config = yaml.safe_load(stream)
        # load shape yaml files.
        dir_files = os.scandir(shapes_dir)
        self.shapes = []

        for shape in shapes_config['dataset_selector']:
            self.shapes.append(shape)

        self.image_size = tuple(shapes_config['image_size'])
        self.min_objects_in_image = shapes_config['min_objects_in_image']
        self.max_objects_in_image = shapes_config['max_objects_in_image']
        self.bg_color = shapes_config['bg_color']
        self.iou_thresh = shapes_config['iou_thresh']
        self.margin_from_edge = shapes_config['margin_from_edge']
        self.bbox_margin = shapes_config['bbox_margin']
        self.size_fluctuation = shapes_config['size_fluctuation']
        self.rotate_shapes = shapes_config['rotate_shapes']

        # create a class names output file:
        self.category_names = [shape['cname'] for shape in self.shapes]
        with open(shapes_config['class_names_file'], 'w') as f:
            for category_name in self.category_names:
                f.write(f'{category_name}\n')

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
        return new_bbox


    def rotate(self):
        val = math.pi/4*np.random.randint(0, 8) if self.rotate_shapes else 0
        return val

    def __create_polygon(self, nvertices, shape_width_choices, shape_aspect_ratio, size_fluctuation, margin_from_edge, image_size):

        """
        Description: Creates a polygon given nvertices, and data on shape's dims
        :param nvertices: type: string name of a supported shape
        :param shape_width_choices: A list of widths choices for random selection.
        :type shape_width_choices: floats list
        :param shape_aspect_ratio: ratio between shapes height and width
        :type shape_aspect_ratio:
        :param size_fluctuation: fluctuations of new bbox dims, each multiplied by (1-rand(0, size_fluctuation)) where
        <=0size_fluctuation<1
        :type size_fluctuation:
        :param margin_from_edge: Minimal distance in pixels between bbox and image edge.
        :type margin_from_edge: int

        :param image_size: Canvas size of target image
        :type image_size: 2 tuple, ints
        :param x_min: type: float. x_min coordinate of the bbox
        :param y_min: type: float. y_min coordinate of the bbox
        :param x_max: type: float. x_max coordinate of the bbox
        :param y_max: type: float. y_max coordinate of the bbox
        :return:
        polygon: type:float. a list of n 2 tuples, where n is the num of polygon vertices and tuples hold x,y coords
        """

        shape_width = np.random.choice(shape_width_choices)
        shape_height = shape_width * shape_aspect_ratio * random.uniform(1 - size_fluctuation, 1)
        # add fluctuations - config defuned
        shape_width = shape_width * random.uniform(1 - size_fluctuation, 1)

        radius = np.array([shape_width / 2, shape_height / 2])
        center = np.random.randint(
            low=radius + margin_from_edge, high=np.floor(image_size - radius - margin_from_edge), size=2)

        polygon = [
            (math.cos(th) * radius[0] + center[0],
             math.sin(th
                      ) * radius[1] + center[1])
            for th in [i * (2 * math.pi) / nvertices for i in range(nvertices)]
        ]

        return polygon

    def __create_ds_entry(self, objects_attributes, image_size, bg_color, iou_thresh,
                          margin_from_edge,
                          bbox_margin,
                          size_fluctuation
                          ):
        """
        Description: Create a single dataset entry, consists of an of shape objects, along with bbox and polygons
        objects. THe latters a . Store created images in output_dir, and return dataset metadata.

        :param objects_attributes: a list of num_of_objects entries with attributes: id,cname, shape_aspect_ratio,
        shape_width_choices, fill_color

        :param image_size: type: 2 tuple of ints. required entry's image size.
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

        for entry_id, nvertices, category_name, shape_aspect_ratio, shape_width_choices, color in objects_attributes:
            max_count = 10000
            count = 0
            # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
            while True:

                polygon = self.__create_polygon(nvertices, shape_width_choices, shape_aspect_ratio, size_fluctuation,
                                                margin_from_edge,
                                                image_size)
                bbox = self.__polygon_to_box(polygon)

                iou = list(map(lambda x: self.__compute_iou(bbox, x), bboxes))

                if len(iou) == 0 or max(iou) <= iou_thresh:
                    break
                if count > max_count:
                    max_iou = max(iou)
                    raise Exception(
                        f'Shape Objects Placement Failed after {count} placement itterations: max(iou)={max_iou}, '
                        f'but required iou_thresh is {iou_thresh} shape_width: . \nHint: reduce objects size or'
                        f' quantity of objects in an image')
            if len(bbox):
                bboxes.append(bbox)

            # draw shape on image:
            draw.polygon(polygon, fill=ImageColor.getrgb(color) )

            polygons.append(polygon)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(entry_id)

        bboxes = np.array(bboxes)
        # transfer bbox coordinate to:  [xmin, ymin, w, h]: (bbox_margin is added distance between shape and bbox)
        bboxes = [bboxes[:, 0] - bbox_margin,
                  bboxes[:, 1] - bbox_margin,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return image, bboxes, tuple(objects_categories_indices), objects_categories_names, polygons


    def create_dataset(self,  nentries, output_dir):
        """
        Description: Create nentries dataset. Store created images in output_dir, and return dataset metadata.

        :param nentries: type: int, number of entries to produce
        :param output_dir: type: str,  destination output dir for dataset's images
        :return:
        images_filenames: type: list of str size:  nentries. created images filenames, w/o dir prefix
        images_sizes: type:  list of 2 tuples ints.  size:  nentries. (image.height, image.width)
        images_bboxes: type:  list of float [nobjects, 4] arrays . size:  nentries. Bounding boxes of image's nobjects
        images_objects_categories_indices: type: list of nobjects tuples size: nentries. Category id of image's nobjects
        self.category_names: type: list of str. size: ncategories. Created dataset's num of categories.
        polygons: type: float. list of nobjects, each object shape with own 2 points nvertices. Needed for segmentation

        """

        images_filenames = []
        images_sizes = []
        images_bboxes = []
        images_objects_categories_indices = []
        images_objects_categories_names = []
        images_polygons = []
        for example_id in range(nentries):
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)
            # randomly select num_of_objects shapes:
            sel_shape_entris = [np.random.choice(self.shapes) for idx in range(num_of_objects)]
            # arrange target objects attributes from selected shapes:
            objects_attributes = [
                [shape_entry['id'], shape_entry['nvertices'], shape_entry['cname'], shape_entry['shape_aspect_ratio'],
                 shape_entry['shape_width_choices'],
                 shape_entry['color']] for shape_entry in sel_shape_entris]
            try:
                image, bboxes, objects_categories_indices, objects_categories_names, polygons = self.__create_ds_entry(
                    objects_attributes,
                    self.image_size,
                    self.bg_color,
                    self.iou_thresh,
                    self.margin_from_edge,
                    self.bbox_margin,
                    self.size_fluctuation)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)

                msg = str(e)
                raise Exception(f'Error type: {exc_type}, file: {fname},{exc_tb.tb_lineno}, Desc: {msg}')
            image_filename = f'img_{example_id + 1:06d}.jpg'
            file_path = f'{output_dir}/images/{image_filename}'
            image.save(file_path)
            if len(bboxes) == 0:
                continue

            images_filenames.append(image_filename)
            images_sizes.append((image.height, image.width))
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)
            images_polygons.append(polygons)

        return images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, self.category_names, images_polygons
