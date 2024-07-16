import numpy as np
import math
from math import cos, sin
import random

class CreatePolygons:
    @property
    def categories_names(self):
        return self._categories_names
    @property
    def shapes_nvertices(self):
        return self._shapes_nvertices

    def __init__(self, config):

        # load shape yaml files.
        shapes_config = config['shapes_config']
        self.image_size = config['image_size']
        self.min_objects_in_image = config['min_objects_in_image']
        self.max_objects_in_image = config['max_objects_in_image']
        # self.iou_thresh = config['iou_thresh']
        self.margin_from_edge = config['margin_from_edge']
        # self.bbox_margin = config['bbox_margin']
        self.size_fluctuation = config['size_fluctuation']

        # create a class names output file.
        self.shapes = []
        for shape in shapes_config:
            self.shapes.append(shape)

        self._categories_names = [shape['cname'] for shape in self.shapes if shape['active']]
        self._shapes_nvertices = [shape['nvertices'] for shape in self.shapes if shape['active']]

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

    def __create_one_image_polygons(self, objects_attributes, image_size,
                          margin_from_edge,
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

        polygons=[]
        objects_categories_names = []
        objects_categories_indices = []
        objects_colors = []
        obb_thetas = []


        for category_id, nvertices, theta0, category_name, aspect_ratio, height, color, obb_theta in objects_attributes:
            polygon = self.__create_polygon(nvertices,theta0, height, aspect_ratio, size_fluctuation,
                                                margin_from_edge,
                                                image_size)


            polygons.append(polygon)
            objects_categories_names.append(category_name)
            objects_categories_indices.append(category_id)
            objects_colors.append(color)
            obb_thetas.append(obb_theta)


        return  tuple(objects_categories_indices), objects_categories_names, polygons, objects_colors, obb_thetas


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


    def create_batch_polygons(self, nentries):
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
        batch_image_size=[]
        batch_objects_categories_indices = []
        batch_objects_categories_names = []
        batch_polygons = []
        batch_objects_colors = []
        batch_obb_thetas = []
        # loop to create nentries examples:
        for example_id in range(nentries):
            # randomize num of objects in an image:
            num_of_objects = np.random.randint(self.min_objects_in_image, self.max_objects_in_image + 1)
            # take only active shapes for dataset creation:
            active_shapes = [shape for shape in self.shapes if shape['active']]
            # randomly select num_of_objects shapes:
            sel_shape_entris = [np.random.choice(active_shapes) for idx in range(num_of_objects)]

            sel_index = random.randint(0, len(self.image_size) - 1)  # randomly select img size index from config
            image_size = self.image_size[sel_index]
            # arrange target objects attributes from selected shapes:
            objects_attributes = [
                [self.categories_names.index(shape_entry['cname']), shape_entry['nvertices'], shape_entry['theta0'],
                 shape_entry['cname'], shape_entry['aspect_ratio'],
                 shape_entry['height'],
                 shape_entry['color'], shape_entry['obb_theta']] for shape_entry in sel_shape_entris]

            objects_categories_indices, objects_categories_names, polygons, objects_colors, obb_thetas = self.__create_one_image_polygons(
                objects_attributes,
                image_size,
                self.margin_from_edge,
                self.size_fluctuation)


            # if len(bboxes) == 0:
            #     continue

            # batch_bboxes.append(bboxes)
            # batch_bboxes.append(bboxes)
            batch_image_size.append(image_size)
            batch_objects_categories_indices.append(objects_categories_indices)
            batch_objects_categories_names.append(objects_categories_names)
            batch_polygons.append(polygons)
            batch_objects_colors.append(objects_colors)
            batch_obb_thetas.append(obb_thetas)
        return batch_image_size, batch_objects_categories_indices, batch_objects_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas

