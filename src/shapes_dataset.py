import numpy as np
import math
from math import cos, sin
import random

class ShapesDataset:
    """
    A class to generate shapes dataset, based on a config file:

    Public method: create_dataset

    """

    def __init__(self, config):
        # with open(shapes_config_file, 'r') as stream:
        #     shapes_config = yaml.safe_load(stream)
        # load shape yaml files.
        shapes_config = config['shapes_config']
        self.image_size = config['image_size']
        self.min_objects_in_image = config['min_objects_in_image']
        self.max_objects_in_image = config['max_objects_in_image']
        self.iou_thresh = config['iou_thresh']
        self.margin_from_edge = config['margin_from_edge']
        self.bbox_margin = config['bbox_margin']
        self.size_fluctuation = config['size_fluctuation']

        # create a class names output file.
        self.shapes = []
        for shape in shapes_config:
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


    def __rotate(self, polygon, theta0):
        """
        Rotates image - wip
        :param polygon:
        :type polygon:
        :param theta0:
        :type theta0:
        :return:
        :rtype:
        """

        rot_angle = theta0/180*math.pi # rot_tick*np.random.randint(0, 8)
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


        polygon = [
            (cos(th) * radius[0],
             sin(th
                      ) * radius[1])
            for th in [i * (2 * math.pi) / nvertices for i in range(nvertices)]
        ]

        # rotate shape:
        if theta0:
            polygon= self.__rotate(polygon, theta0)

        # translate to center:
        center = np.random.randint(
            low=radius + margin_from_edge, high=np.floor(image_size - radius - margin_from_edge), size=2)
        polygon+=center
        # polygon=tuple(map(tuple, polygon))
        return polygon


    def __create_one_image_shapes(self, objects_attributes, image_size, iou_thresh,
                          margin_from_edge,
                          bbox_margin,
                          size_fluctuation
                          ):
        """
        Create a single dataset entry, with nt bboxes and polygons.

        :param objects_attributes: list[nt], entry: [cls_id, nvertices, theta0, cls_name, [aspect_ratio], [height],
        [color]] where aspect ratio, height and color(str) are lists for random selection,
        :param image_size: image size, list[2] height and width
        :param iou_thresh: type: float [0,1], maximal iou value for adjacent bboxes. iou=1 means complete overlap. iou=0 means no overlap
        :param margin_from_edge: type: int. minimal distance in pixels of bbox from image's edge.
        :param bbox_margin: type: int. distance in pixels between bbox and shape
        :param size_fluctuation: int float [0,1), images' width and height are multiplied by (1-rand(size_fluctuation))
        :return:
            image: an RGB pillow image drawn with shapes
            bboxes: type: float ndarray [nobjects,4]  [xc, yc, w, h]
            tuple(objects_categories_indices): type in. tuple of nobjects category indices
            objects_categories_names: type: str. list of nobjects category names
            polygons: type: float. list of nobjects, each object shape with own 2 points nvertices
        """

        bboxes = []
        polygons=[]
        objects_categories_names = []
        objects_categories_indices = []
        objects_colors = []
        obb_thetas = []


        for category_id, nvertices, theta0, category_name, aspect_ratio, height, color, obb_theta in objects_attributes:
            max_count = 10
            count = 0
            # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
            # while True:
            polygon = self.__create_polygon(nvertices,theta0, height, aspect_ratio, size_fluctuation,
                                                margin_from_edge,
                                                image_size)
            bbox = self.__polygon_to_box(polygon)


            iou = list(map(lambda x: self.__compute_iou(bbox, x), bboxes)) # check iou with other generated boxes, must be below thresh
            if len(iou) !=0 and np.any(np.array(iou)) > iou_thresh:
                print(f'\nDropping shape polygon from image: iou above theshold {category_id} {nvertices} {category_name} {aspect_ratio} {height}')
                continue
            if np.any(np.array(bbox) > image_size[0]) or np.any(np.array(bbox) < 0): # debug!!!
                print(f'\nDropping shape polygon from image: shape exceeds image boundaries {category_id} {nvertices} {category_name} {aspect_ratio} {height}')
                continue
            if len(bbox):
                bboxes.append(bbox)


            polygons.append(polygon)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(category_id)
            objects_colors.append(color)
            obb_thetas.append(obb_theta)


        bboxes = np.array(bboxes)
        # [x,y,x,y] to [xc, yc, w, h] (bbox_margin is added distance between shape and bbox):
        bboxes = [(bboxes[:, 0] +  bboxes[:, 2])/2,
                  (bboxes[:, 1] +  bboxes[:, 3])/2,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return bboxes, tuple(objects_categories_indices), objects_categories_names, polygons, objects_colors, obb_thetas


    def create_images_shapes(self,  nentries):
        """
        Description: Create dataset entries. svae created images in output_dir, and return dataset metadata.

        :param nentries: type: int, number of entries to produce
        :param output_dir: type: str,  destination output dir for dataset's images
        :return:
        batch_filenames: type: list of str size:  nentries. created images filenames, w/o dir prefix
        batch_sizes: type:  list of 2 tuples ints.  size:  nentries. (image.height, image.width)
        batch_bboxes: type:  list of float [nobjects, 4] arrays . size:  nentries. Bounding boxes of image's nobjects  [xc, yc, w, h]
        batch_objects_categories_indices: type: list of nobjects tuples size: nentries. Category id of image's nobjects
        self.category_names: type: list of str. size: ncategories. Created dataset's num of categories.
        self.category_ids:  type: list of int. size: ncategories. Created dataset's entries ids.
        polygons: type: float. list of nobjects, each object shape with own 2 points nvertices. Needed for segmentation

        """

        batch_bboxes = []
        batch_objects_categories_indices = []
        batch_objects_categories_names = []
        batch_polygons = []
        batch_objects_colors=[]
        batch_obb_thetas=[]
        # loop to create nentries examples:
        for example_id in range(nentries):
            # randomize num of objects in an image:
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)
            # take only active shapes for dataset creation:
            active_shapes = [shape   for shape in self.shapes if shape['active']]
            # randomly select num_of_objects shapes:
            sel_shape_entris = [np.random.choice(active_shapes) for idx in range(num_of_objects)]

            sel_index = random.randint(0, len(self.image_size)-1) # randomly select img size index from config
            image_size=self.image_size[sel_index]
            # arrange target objects attributes from selected shapes:
            objects_attributes = [
                [self.category_names.index(shape_entry['cname']), shape_entry['nvertices'], shape_entry['theta0'], shape_entry['cname'], shape_entry['aspect_ratio'],
                 shape_entry['height'],
                 shape_entry['color'], shape_entry['obb_theta']] for shape_entry in sel_shape_entris]

            bboxes, objects_categories_indices, objects_categories_names, polygons, objects_colors, obb_thetas = self.__create_one_image_shapes(
                    objects_attributes,
                    image_size,
                    self.iou_thresh,
                    self.margin_from_edge,
                    self.bbox_margin,
                    self.size_fluctuation)

            if len(bboxes) == 0:
                continue

            batch_bboxes.append(bboxes)
            batch_objects_categories_indices.append(objects_categories_indices)
            batch_objects_categories_names.append(objects_categories_names)
            batch_polygons.append(polygons)
            batch_objects_colors.append(objects_colors)
            batch_obb_thetas.append(obb_thetas)
        return batch_bboxes, batch_objects_categories_indices, batch_objects_categories_names, self.category_names, self.category_ids, batch_polygons, batch_objects_colors, batch_obb_thetas
