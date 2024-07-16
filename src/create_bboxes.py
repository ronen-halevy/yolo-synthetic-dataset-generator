import numpy as np

class CreateBboxes:
    def __init__(self, iou_thresh, bbox_margin):
        self.iou_thresh = iou_thresh
        self.bbox_margin = bbox_margin

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

    def __create_one_image_bboxes(self,
                                  polygons,
                                  iou_thresh,
                                  bbox_margin,
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
        for polygon in polygons:
            max_count = 10
            count = 0
            # Iterative loop to find location for shape placement i.e. center. Max iou with prev boxes should be g.t. iou_thresh
            # while True:

            bbox = self.__polygon_to_box(polygon)


            # iou = list(map(lambda x: self.__compute_iou(bbox, x), bboxes)) # check iou with other generated boxes, must be below thresh
            # if len(iou) !=0 and np.any(np.array(iou)) > iou_thresh:
            #     print(f'Dropping shape polygon from image: iou above theshold {category_name} height: {height} aspect_ratio: {aspect_ratio} ')
            #     continue
            # if np.any(np.array(bbox) > image_size[0]) or np.any(np.array(bbox) < 0): # debug!!!
            #     print(f'Dropping shape polygon from image: shape exceeds image boundaries  {category_name} height: {height} aspect_ratio: {aspect_ratio}')
            #     continue
            # if len(bbox):
            bboxes.append(bbox)


            # polygons.append(polygon)
            # objects_categories_names.append(category_name)
            # objects_categories_indices.append(category_id)
            # objects_colors.append(color)
            # obb_thetas.append(obb_theta)


        bboxes = np.array(bboxes)
        # [x,y,x,y] to [xc, yc, w, h] (bbox_margin is added distance between shape and bbox):
        bboxes = [(bboxes[:, 0] +  bboxes[:, 2])/2,
                  (bboxes[:, 1] +  bboxes[:, 3])/2,
                  bboxes[:, 2] - bboxes[:, 0] + 2 * bbox_margin,
                  bboxes[:, 3] - bboxes[:, 1] + 2 * bbox_margin]

        bboxes = np.stack(bboxes, axis=1)  # / np.tile(image_size, 2)

        return bboxes

    def normalize_bboxes(self, images_bboxes, images_sizes):
        """

        Description: one *.txt file per image,  one row per object, row format: class x_center y_center width height.
        normalized coordinates [0 to 1].
        zero-indexed class numbers - start from 0

        :param images_paths: list of dataset image filenames
        :param images_bboxes: list of per image bboxes arrays in  [xc,yc,w,h] format.
        :param images_sizes:
        :param images_class_ids:  list of per image class_ids arrays
        :param output_dir: output dir of labels text files
        :return:
        """

        all_bboxes = []
        for bboxes, images_size,  in zip(images_bboxes, images_sizes
                                                   ):  # images loop
            im_height = images_size[0]
            im_width = images_size[1]

            # head, filename = os.path.split(image_path)
            bboxes = np.array(bboxes, dtype=float)
            img_bboxes = []

            for bbox in bboxes:  # labels in image loop
                # normalize scale:
                xywh_bbox = [bbox[0] / im_width, bbox[1] / im_height,
                             bbox[2] / im_width, bbox[3] / im_height]

                entry = f"{' '.join(str(e) for e in xywh_bbox)}"
                img_bboxes.append(entry)
            all_bboxes.append(img_bboxes)
        return all_bboxes


    def create_batch_bboxes(self,  batch_polygons, batch_image_size):
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
        # batch_objects_categories_indices = []
        # batch_objects_categories_names = []
        # batch_polygons = []
        # batch_objects_colors=[]
        # batch_obb_thetas=[]
        # loop to create nentries examples:
        for img_polygons in batch_polygons:
            bboxes = self.__create_one_image_bboxes(
                    img_polygons,
                    self.iou_thresh,
                    self.bbox_margin,
            )

            if len(bboxes) == 0:
                continue

            batch_bboxes.append(bboxes)
            batch_bboxes_normed = self. normalize_bboxes(batch_bboxes, batch_image_size)
        return batch_bboxes_normed
