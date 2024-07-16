import numpy as np
import math
from src.create.create_polygons import CreatePolygons
from src.create.create_bboxes import CreateBboxes
from src.create.utils import calc_iou

def xywh2xyxy(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    center, w, h = np.split(obboxes, (2, 3), axis=-1)

    point1 = center + np.concatenate([-w/2,-h/2], axis=1)
    point2 = center + np.concatenate([w/2,-h/2], axis=1)
    point3 = center + np.concatenate([w/2,h/2], axis=1)
    point4 = center + np.concatenate([-w/2,h/2], axis=1)

    # order = obboxes.shape[:-1]
    return np.concatenate(
            [point1, point2, point3, point4], axis=-1)

def rotate(hbboxes, theta0):
    rot_angle = np.array(theta0) / 180 * math.pi  # rot_tick*np.random.randint(0, 8)

    rotate_bbox = lambda xy: np.concatenate([np.sum(xy * np.concatenate([np.cos(rot_angle)[...,None, None], np.sin(rot_angle)[...,None, None]], axis=-1), axis=-1,keepdims=True),
                              np.sum(xy * np.concatenate([-np.sin(rot_angle)[...,None, None], np.cos(rot_angle)[...,None, None]], axis=-1), axis=-1,keepdims=True)], axis=-1)
    offset_xy = (np.max(hbboxes, axis=-2, keepdims=True) + np.min(hbboxes,axis=-2, keepdims=True)) / 2
    hbboxes_ = hbboxes - offset_xy # remove offset b4 rotation
    rbboxes =  rotate_bbox(hbboxes_)
    rbboxes=rbboxes+offset_xy # add offset back
    return rbboxes


def create_obb_entries(bbox_entries):
    """

    :param bbox_entries:
    :type bbox_entries:
    :return:
    :rtype:
    """
    bboxes = []
    for idx, bbox_entry in enumerate(bbox_entries):  # loop on images
        bbox_entries = xywh2xyxy(bbox_entry)
        bboxes.append(bbox_entries)
    return bboxes


def arrange_obb_entries(images_polygons, images_size, categories_lists):
    batch_entries = []
    for image_polygons, image_size, class_ids in zip(images_polygons, images_size,
                                                     categories_lists):
        # normalize sizes:
        image_polygons = [image_polygon / np.array(image_size) for image_polygon in image_polygons]

        image_entries = [
            f"{category_id} {' '.join(str(vertix) for vertix in list(image_polygon.reshape(-1)))}\n" for
            image_polygon, category_id in zip(image_polygons, class_ids)]
        batch_entries.append(image_entries)
    return batch_entries


def remove_dropped_bboxes(batch_bbox_entries, dropped_ids):
    """
    Filter bbox entries by dropping entries according to dropped_ids vector
    :param batch_bbox_entries:list[batch][nti][5] where nti is nof bbox in image i, each entry is cls,bbox, normalized
    :param dropped_ids: list[nd], each entry holds id of to-be-droped bbox: [img_id,bbox_id]
    :type dropped_ids:
    :return:
    batch_bbox_entries_filtered: list[batch][nti_filtered][5]
    """
    batch_bbox_entries_filtered = []
    for img_idx, img_bbox_entry in enumerate(batch_bbox_entries,):  # loop on images
        img_bbox_drop_ids = [drop_id[1] for drop_id in dropped_ids if drop_id[0] == img_idx]
        img_bbox_filtered_entries = np.delete(np.array(img_bbox_entry), img_bbox_drop_ids, axis=0)
        batch_bbox_entries_filtered.append(img_bbox_filtered_entries)
    return batch_bbox_entries_filtered

def filter_polygons(batch_polygons, batch_filters):
    """
    Filter polygons by bool filters
    :param batch_polygons: list[batch][nti] of [nvi,2], where nvj is nof vertices in polygon j of image i. (non normed)
    :param filter: bool, list[batch] of array(nti), filter polygons
    :return:
    batch_polygons_filtered:  list[batch][nti_f] of [nvi,2], nvj is nof vertices in polygon j of image i.(non normed)

    """
    batch_polygons_filtered = []
    for img_idx, (img_polygons, img_filters)  in enumerate(zip(batch_polygons,batch_filters)):  # loop on images
        img_polygons_filtered = [polygon for polygon, filter in zip(img_polygons, img_filters) if filter]
        batch_polygons_filtered.append(img_polygons_filtered)
    return batch_polygons_filtered

def rotate_polygon_entries(batch_polygons, images_sizes, batch_thetas, iou_thresh=0):
    """
    Rotate polygons by theta.
    If a rotated polygon crosses image boundaries, keep unrotated polygon.
    If iou between a rotated polygon and any polygon in list crosses iou_thresh, then leave unrotated (set theta to 0)
    If iou of unrotated polygon and any polygon in list still crosses iou_thresh, then drop polygon.
    :param batch_polygons: batches polygons list. list size: [bimgs, npolygons], np.array polygons shape: [nvertices,2]
    :param images_size: tuple, [img_w, img_h], used for rotated polygon boundary check
    :param theta: polygons rotation angle in degrees.
    :param iou_thresh: max permitted iou between any image's pair of polygons
    :return:
       batches rpolygons: rotated polygons list. size: [bimgs, npolygons], entry: np.array polygons shape: [nvertices,2]
       batch_result_thetas: actual thetas list. size: [bimgs, npolygons], entry: float/int
       dropped_ids: dropped polygons, (due to iou above thresh). ids list. size: [bimgs, npolygons], entry: int
    :rtype:
    """
    batch_rpolygons = []
    batch_result_thetas = []
    dropped_ids=[]
    # loop on batch images:
    for im_idx, (image_polygons, image_size, thetas) in enumerate(zip(batch_polygons, images_sizes, batch_thetas)):
        rpolygons=[]
        res_thetas=[]
        for idx, (polygon, theta) in enumerate(zip(image_polygons, thetas)): # loop on image's polygons
            unrotate = False # reset unrotate fkag
            rpolygon = rotate(polygon, theta)
            # check if rotated shape is inside image bounderies, otherwise leave unrotated:
            if np.any(rpolygon > image_size) or np.any(rpolygon < 0):
                rpolygon=polygon # replace rotated by original unrotated
                unrotate = True
                print(f'\n Rotated shape is outsode image  boundaries. Keep unrotate. img id: {im_idx} shape id: {idx}')
            # if of rotated with already rotated list:
            if np.any(calc_iou(rpolygon, rpolygons) > iou_thresh):
                # iou of rotated above thresh, so either keep unrotated or drop if iou above thresh:
                if np.any(calc_iou(polygon, rpolygons) > iou_thresh):# iou for unrotated: either drop or keep unrotated
                    dropped_ids.append([im_idx, idx])
                    print(f'IOU with rotated images exceeds treshs: Droppng  img_id: {im_idx} shape_id: {idx}')
                    continue
                else:
                    print(f'IOU of unrotated passed. Keep unrotated shape. img_id {im_idx} shape_id: {idx}')
                    unrotate = True
                    rpolygon=polygon

            rpolygons.append(rpolygon)
            if unrotate:
                res_thetas.append(0)
            else:
                print(f'Rotate shape by {theta} degrees. img_id: {im_idx} shape_id: {idx} ')
                res_thetas.append(theta)

        batch_result_thetas.append(res_thetas)
        batch_rpolygons.append(rpolygons)
    print(f'Batch rotations angles: {batch_result_thetas}')
    return batch_rpolygons, batch_result_thetas, dropped_ids


def rotate_obb_bbox_entries(batch_bboxes, images_size, obb_thetas):
    """
    Rotate bboxes, filter only if inside image boundaries, return in bounderies filter to filter related polygonsL
    :param batch_bboxes: list[batch[ of arrays[nti,8] where nti is nof bbox in image i, 8 are 4 xy bbox coords, 1 normed
    :param images_size: list[batch] of array[2], used to check image limits crossing
    :param obb_thetas: liat[bath] of list[nti] where nti is nof bbox in image i.
    :return:
    batch_rbboxes: list[batch] of array[nti,8]where nti is nof bbox in image i, normed to 1
    batch_in_boundaries: bool, list[batch] of array[nti]where nti is nof bbox in image i.
    :rtype:
    """
    batch_rbboxes = []
    batch_in_boundaries=[]
    for im_idx, (img_hbboxes, image_size, theta) in enumerate(zip(batch_bboxes, images_size, obb_thetas)):
        img_rbboxes = rotate(img_hbboxes.reshape([-1, 4, 2]), theta).reshape(-1, 8)
        img_in_bounderies = np.logical_and(img_rbboxes / np.tile(image_size,[4]) < 1, img_rbboxes >  0) # bool, shape[nimg_bboxes, 8]
        img_in_bounderies = np.all(img_in_bounderies, axis=-1) # bool, shape[nimg_bboxes]
        img_rbboxes=img_rbboxes[img_in_bounderies] # filter
        batch_rbboxes.append(img_rbboxes)
        batch_in_boundaries.append(img_in_bounderies)
    return batch_rbboxes, batch_in_boundaries


def append_category_field(batch_rbboxes, batch_objects_categories_names):
    """
    append category name at the end of each entry (as in dota format for obb)

    :param batch_rbboxes: list size: [batch][nti][entry string] where an entry string holds 8 bbox normed coordinates.
    :param batch_objects_categories_names: list of objects' names. list size: [batch][nti], string
    :return: batch_rbboxes_update, each entry apendedd with a category name.  list size: [batch][nti][entry string]
    """
    batch_rbboxes_update = []
    for img_rbboxes, img_objects_categories_names in zip(batch_rbboxes, batch_objects_categories_names):
        img_rbboxes_update = []
        for rbbox, img_object_category_name in zip(img_rbboxes, img_objects_categories_names):
            entry=rbbox.tolist()
            entry.append(img_object_category_name)
            img_rbboxes_update.append(entry)
        batch_rbboxes_update.append(img_rbboxes_update)
    return batch_rbboxes_update

class CreateObbEntries(CreatePolygons, CreateBboxes):
    def __init__(self, config, iou_thresh, bbox_margin):
        CreatePolygons.__init__(self, config)
        CreateBboxes.__init__(self, iou_thresh, bbox_margin)
    def run(self, nentries):
        batch_image_size, batch_categories_ids, batch_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas = self.create_batch_polygons(
            nentries)
        batch_bboxes = self.create_batch_bboxes(batch_polygons, batch_image_size)
        batch_labels, batch_polygons=self.create_obb_labels(batch_polygons, batch_bboxes,   batch_image_size, batch_obb_thetas, batch_categories_names)
        return batch_polygons, batch_labels, batch_objects_colors, batch_image_size

    def create_obb_labels(self, batch_polygons, bbox_entries, images_size, obb_thetas, batch_objects_categories_names):
        batch_polygons, batch_obb_thetas, dropped_ids = rotate_polygon_entries(batch_polygons, images_size, obb_thetas)
        bbox_entries = remove_dropped_bboxes(bbox_entries, dropped_ids)
        bbox_entries = create_obb_entries(bbox_entries)
        batch_rbboxes, batch_in_bounderies = rotate_obb_bbox_entries(bbox_entries, images_size, batch_obb_thetas)
        batch_polygons = filter_polygons(batch_polygons, batch_in_bounderies)

        batch_rbboxes = append_category_field(batch_rbboxes, batch_objects_categories_names)

        def entries_list_to_string(batch_rbboxes):
            batch_rbboxes_strings = []
            for img_rbboxes in batch_rbboxes:
                img_rbboxes = [' '.join(str(x) for x in img_rbboxes[idx]) for idx in range(len(img_rbboxes))]
                batch_rbboxes_strings.append(img_rbboxes)
            return batch_rbboxes_strings

        batch_labels = entries_list_to_string(batch_rbboxes)
        return batch_labels, batch_polygons
