import numpy as np

from src.create.create_polygons import CreatePolygons
from src.create.create_bboxes import CreateBboxes
class CreatesKptsEntries(CreatePolygons, CreateBboxes):
    def __init__(self, config, iou_thresh, bbox_margin):
        CreatePolygons.__init__(self, config)
        CreateBboxes.__init__(self, iou_thresh, bbox_margin)
    def run(self, nentries):
        batch_image_size, batch_categories_ids, batch_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas = self.create_batch_polygons(nentries
            )
        batch_bboxes = self.create_batch_bboxes(batch_polygons)
        batch_bboxes = self.normalize_bboxes(batch_bboxes, batch_image_size)
        batch_labels=self.create_detection_kpts_entries(batch_bboxes, batch_polygons, batch_image_size, batch_categories_ids)
        return batch_polygons, batch_labels, batch_objects_colors, batch_image_size


    def create_detection_kpts_entries(self, batch_bboxes, batch_polygons, batch_img_sizes, batch_categories_ids):
        """

        :param batch_bboxes: list[bsize] of np.array[nti,4] entries, i=0:bsize . float Unormalized. [xc,yc,w,h]
        :type batch_polygons: list[bsize] of list[nti] entries, i=0:bsize, each entry is a polygon np.array[n_vertices_j,2]
        :type batch_img_sizes:  list[bisize] of list[2] entries, each holds image's [w,h]
        :param batch_class_ids: list[bsize] of list[nti], each entry is the related class id.
        :return:
        entries: list[bsize] of string entries: 'xc yc wh kpt0_x kpt0_y occlusion0.......kptnx kptny occlusionn'
        """

        # detection_entries = create_detection_entries(batch_bboxes, batch_img_sizes, batch_class_ids)
        entries=[]

        nkpts = max(self.polygons_nvertices) # polygons with less verticess are 0 padded, for a uniform entry size
        for image_polygons, image_size, image_categories_id, image_bboxes in zip(batch_polygons, batch_img_sizes,
                                                                  batch_categories_ids, batch_bboxes):

            image_bboxes=np.array(image_bboxes)
            im_height = image_size[0]
            im_width = image_size[1]

            # image_bboxes = (image_bboxes/np.array([im_width, im_height, im_width, im_height]))
            img_kpts = []
            for polygon in  image_polygons:
                kpts = polygon/np.array([im_width, im_height])
                pad = np.zeros([(nkpts - kpts.shape[0])*3])
                occlusion_fields = np.full([kpts.shape[0], 1], 2.)
                # concat occlusion  (=valid) field:
                kpts = np.concatenate([kpts, occlusion_fields], axis=-1).reshape([-1])
                kpts=np.append(kpts,pad)
                img_kpts.append(kpts)

            img_entries=[]
            for bbox, kpts, category_id   in zip(image_bboxes, img_kpts, image_categories_id):
                # box = ' '.join(str( round(vertex, 2)) for vertex in list(bbox))
                box = ' '.join(str(round(vertex,3)) for vertex in list(bbox))
                entry = f"{category_id} {box} {' '.join(str( round(kpt, 2)) for kpt in list(kpts.reshape(-1)))}"
                img_entries.append(entry)
            entries.append(img_entries)
        return entries

