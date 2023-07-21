import numpy as np
from PIL import Image, ImageDraw
import math
import random
import yaml
import os
from PIL import Image,ImageColor



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
        self.shapes=[]
        for shape_file in dir_files:
            if shape_file.name.endswith(".yaml"):
                with open(f'{shapes_dir}{shape_file.name}', 'r') as stream:
                    shape = yaml.safe_load(stream)
                # add shape category to dataset if included  by dataset_selector setup:
                if shape['cname'] in shapes_config['dataset_selector'].keys():
                    # assign shape category with an id according to dataset_selector setup:
                    shape.update({'id':  shapes_config['dataset_selector'][shape['cname']]['id']})
                    self.shapes.append(shape)

        self.image_size = tuple(shapes_config['image_size'])
        self.min_objects_in_image = shapes_config['min_objects_in_image']
        self.max_objects_in_image = shapes_config['max_objects_in_image']
        self.bg_color = tuple(shapes_config['bg_color'])
        self.iou_thresh = shapes_config['iou_thresh']
        self.margin_from_edge = shapes_config['margin_from_edge']
        self.bbox_margin = shapes_config['bbox_margin']
        self.size_fluctuation = shapes_config['size_fluctuation']
        self.rotate_shapes=shapes_config['rotate_shapes']

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



    def __create_bbox(self, image_size, bboxes, shape_width_choices, axis_ratio, iou_thresh, margin_from_edge,
                      size_fluctuation=0.01):
        """
        Description: Creates a bbox, randomly placed within image boundaries according to margin_from_edge and iou
        constraint on overlap with other created bboxes. Bbox width and height is according to shape_width_choices,
        axis_ratio and size_fluctuation.

        :param image_size: Canvas size of target image
        :type image_size: 2 tuple, ints
        :param bboxes: a list of already created bboxes, used for iou calc.
        :type bboxes: float
        :param shape_width_choices: A list of widths choices for random selection.
        :type shape_width_choices: floats list
        :param axis_ratio: ratio between shapes height and width
        :type axis_ratio:
        :param iou_thresh: Max permitted iou between new bbox and other already created bbox. iou_thresh=[0,1], where 1
        means fully overlapped.
        :type iou_thresh: float
        :param margin_from_edge: Minimal distance in pixels between bbox and image edge.
        :type margin_from_edge: int
        :param size_fluctuation: fluctuations of new bbox dims, each multiplied by (1-rand(0, size_fluctuation)) where
        <=0size_fluctuation<1
        :type size_fluctuation:
        :return:
        :rtype:
        new_bbox: np array dim [4]. created bbox
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

    def rotate(self):
        val = math.pi/4*np.random.randint(0, 8) if self.rotate_shapes else 0
        return val

    def __create_polygon(self, shape, x_min, y_min, x_max, y_max):
        """
        Description: Creates a polygon given a shape name and a bounding box
        :param shape: type: string name of a supported shape
        :param x_min: type: float. x_min coordinate of the bbox
        :param y_min: type: float. y_min coordinate of the bbox
        :param x_max: type: float. x_max coordinate of the bbox
        :param y_max: type: float. y_max coordinate of the bbox
        :return:
        polygon: type:float. a list of n 2 tuples, where n is the num of polygon vertices and tuples hold x,y coords
        """
        if shape in ['ellipse', 'circle']:
            # draw.ellipse([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)
            n_points=100
            t = np.linspace(0,360,n_points)
            x_coord =(x_min+x_max)/2+(x_max-x_min)/2*np.cos(np.radians(t))
            y_coord = (y_min+y_max)/2+(y_max-y_min)/2*np.sin(np.radians(t+180))
            polygon  = [(x_poly,y_poly) for x_poly,y_poly in zip(x_coord,y_coord)]
        elif shape in ['rectangle', 'square']:
            # draw.rectangle((x_min, y_min, x_max, y_max), fill=fill_color, outline=outline_color, width=3)
            polygon=[ (x_min, y_min),(x_min, y_max), (x_max, y_max), (x_max, y_min)]

        elif shape == 'triangle':
            polygon = [(x_min, y_min), (x_min, y_max), ((x_min + x_max) / 2, y_min)]
            polygon = [x_min, y_max, x_max, y_max, (x_min + x_max) / 2, y_min]

        elif shape in ['trapezoid', 'hexagon',  'rhombus', 'triangle']:
            sides = 3 if shape in ['triangle'] else 4 if shape in  ['parallelogram', 'rhombus'] else 5 if shape == 'trapezoid' else 6
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            rad_x, rad_y = (x_max - x_min) / 2, (y_max - y_min) / 2
            rot_angle=self.rotate()
            polygon= [
                (math.cos(th+rot_angle) * rad_x + center_x,
                 math.sin(th+rot_angle) * rad_y + center_y)
                for th in [i * (2 * math.pi)/ sides  for i in range(sides)]
            ]

        else:
            raise Exception(f'Error: shape {shape} undefined. terminating')
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
        shape_width_choices, fill_color,outline_color

        :param image_size: type: 2 tuple of ints. required entry's image size.
        :param bg_color: image's bg color
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

        for entry_id, category_name, shape_aspect_ratio, shape_width_choices, color, outline_color in objects_attributes:
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
            x_min, y_min, x_max, y_max = bbox.tolist()
            polygon = self.__create_polygon(category_name, x_min, y_min, x_max, y_max)
            # draw shape on image:
            draw.polygon(polygon, fill=ImageColor.getrgb(color), outline=outline_color)


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
        images_polygons=[]
        for example_id in range(nentries):
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)
            # randomly select num_of_objects shapes:
            sel_shape_entris= [np.random.choice(self.shapes) for idx in range(num_of_objects)]
            # arrange target objects attributes from selected shapes:
            objects_attributes = [[shape_entry['id'],  shape_entry['cname'], shape_entry['shape_aspect_ratio'], shape_entry['shape_width_choices'],
                                 shape_entry['color'], tuple(shape_entry['outline_color'])] for shape_entry in sel_shape_entris]
            try:
                image, bboxes, objects_categories_indices, objects_categories_names, polygons = self.__create_ds_entry(objects_attributes,
                                                                                                             self.image_size,
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
            images_sizes.append((image.height, image.width))
            images_bboxes.append(bboxes)
            images_objects_categories_indices.append(objects_categories_indices)
            images_objects_categories_names.append(objects_categories_names)
            images_polygons.append(polygons)


        return images_filenames, images_sizes, images_bboxes, images_objects_categories_indices, self.category_names, images_polygons