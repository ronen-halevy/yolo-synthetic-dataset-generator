import json
from datetime import date, datetime
import numpy as np
import os
import yaml
import math
from shapely.geometry import Polygon


def segments2bboxes_batch(segments, width=640, height=640):
    """
    Convert segment polygons to bounding boxes labels, applying inside-image constraint.

    :param segments: shape: [nobjects, nvertices, 2]
    :type segments:
    :param width:
    :type width:
    :param height:
    :type height:
    :return:
    :rtype:
    """
    # 1. Locate out of region entries, i.e. entries with negative or above image dimenssions coords.

    ge = np.logical_or(np.less(segments[...,0:1], 0), np.less(segments[...,1:2], 0))
    le = np.logical_or(np.greater(segments[...,0:1], width), np.greater(segments[...,1:2], height))
    out_of_region = np.logical_or(ge, le).astype(np.float32) # values 0 or 1, shape: [nt, nvertices, 1]
    # 2. Find bbox xmin,ymin coords
    # 2.1 De-priorities selection of Negative out-of-region coords as xmin, ymin, by adding a large bias
    bias = 10000
    bias_vector = out_of_region*bias # bias is 0 for in region coords, and large otherwise.
    segments_x = segments[..., 0] #+ bias_vector # Add large bias to out of range x coords.
    segments_y = segments[..., 1] #+ bias_vector # Add large bias to out of range y coords.
    # 2.2 find xmin, ymin

    xmin= np.min(segments_x,axis=1)
    ymin = np.min(segments_y, axis=1)
    # 3. Find bbox max coords
    # 3.1 De-priorities selection of positive out-of-region coords xmax, ymax, by substractinb a large bias:
    # segments_x = segments[..., 1]# - 2*bias_vector # substact large bias to out of range x coords.
    # segments_y = segments[..., 2] #- 2*bias_vector # substact large bias to out of range y coords.
    # 3.2 find max coords:
    xmax= np.max(segments_x,axis=1)
    ymax = np.max(segments_y, axis=1)
    # 4 concat bboxes:
    bbox = np.concatenate([xmin[...,None], ymin[...,None], xmax[...,None], ymax[...,None]], axis=-1) # shape: [nt,4]
    # 5 handle edge case of all segment's vertices out of region, which led to biased vertices selection. set 0s bbox:

    ind = np.logical_and(np.greater(bbox, 0), np.less(bbox, bias/2)) # thresh at bias/2 should be good
    bbox = np.where(ind, bbox, [0., 0., 0., 0.]) # if all segments are out of region, then set bbox to 0s
    return bbox


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


def calc_iou(polygon1, polygons):
    """
    Calc iou between polygon1 and polygons - a list of polygons
    :param polygon: polygon vertices, np.array, shape: [nvertices,2]
    :param polygons: a list of np,array polygons of shape [nvertices,2]
    :return: a list, iou of polygon1 and polygons
    """
    polygon1 = Polygon(polygon1)
    iou=[]
    for polygon2 in polygons:
        polygon2 = Polygon(polygon2)
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou.append(intersect / union)
    return np.array(iou)


def remove_dropped_bboxes(bbox_entries, dropped_ids):
    bbox_entries_modified = []
    for idx, bbox_entry in enumerate(bbox_entries,):  # loop on images
        ii = [ad[1] for ad in dropped_ids if ad[0] == idx]  # todo deleted here!!!!
        bbox_entry1 = np.delete(np.array(bbox_entry), ii).tolist()
        bbox_entries_modified.append(bbox_entry1)
    return bbox_entries_modified


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
            # if of rotated with already rotated list: if above thresh, keep unrotated or drop if iou above thresh:
            if np.any(calc_iou(rpolygon, rpolygons) > iou_thresh):
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


def rotate_obb_bbox_entries(bbox_entries, images_size, obb_thetas):
    batch_rbboxes = []
    for hbboxes, image_size, theta in zip(bbox_entries, images_size, obb_thetas):
        hbboxes[:, :8] = rotate(hbboxes[:, :8].reshape([-1, 4, 2]), theta).reshape(-1, 8)
        rbboxes = [' '.join(str(x) for x in hbboxes[idx]) for idx in
                   range(len(hbboxes))]  # store string entries
        batch_rbboxes.append(rbboxes)
    return batch_rbboxes
def arrange_segmentation_entries(images_polygons, images_size, categories_lists):
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


def create_detection_labels_unified_file(images_paths, images_bboxes, images_class_ids,
                                labels_file_path):
    """
    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_class_ids:  list of per image class_ids arrays
    :param labels_file_path: path of output labels text files
    :return:
    """
    print('create_detection_labels_unified_file')

    entries = []
    for image_path, class_ids, bboxes in zip(images_paths, images_class_ids, images_bboxes):

        entry = f'{image_path} '
        for bbox, category_id in zip(bboxes, class_ids):
            bbox_arr = np.array(bbox, dtype=float)
            xyxy_bbox = [bbox_arr[0], bbox_arr[1], bbox_arr[0] + bbox_arr[2], bbox_arr[1] + bbox_arr[3]]
            for vertex in xyxy_bbox:
                entry = f'{entry}{vertex},'
            entry = f'{entry}{float(category_id)} '
        entries.append(entry)
        file = open(labels_file_path, 'w')
        for item in entries:
            file.write(item + "\n")
    file.close()

def xywh2xyxy(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    cls,  center, w, h = np.split(obboxes, (1, 3, 4), axis=-1)

    point1 = center + np.concatenate([-w/2,-h/2], axis=1)
    point2 = center + np.concatenate([w/2,-h/2], axis=1)
    point3 = center + np.concatenate([w/2,h/2], axis=1)
    point4 = center + np.concatenate([-w/2,h/2], axis=1)

    # order = obboxes.shape[:-1]
    return np.concatenate(
            [point1, point2, point3, point4, cls], axis=-1)

def rotate(hbboxes, theta0):
    rot_angle = np.array(theta0) / 180 * math.pi  # rot_tick*np.random.randint(0, 8)

    rotate_bbox = lambda xy: np.concatenate([np.sum(xy * np.concatenate([np.cos(rot_angle)[...,None, None], np.sin(rot_angle)[...,None, None]], axis=-1), axis=-1,keepdims=True),
                              np.sum(xy * np.concatenate([-np.sin(rot_angle)[...,None, None], np.cos(rot_angle)[...,None, None]], axis=-1), axis=-1,keepdims=True)], axis=-1)
    offset_xy = (np.max(hbboxes, axis=-2, keepdims=True) + np.min(hbboxes,axis=-2, keepdims=True)) / 2
    hbboxes_ = hbboxes - offset_xy
    rbboxes =  rotate_bbox(hbboxes_)
    rbboxes=rbboxes+offset_xy
    return rbboxes


def create_obb_entries(bbox_entries):
    bboxes = []
    for idx, bbox_entry in enumerate(bbox_entries):  # loop on images
        bbox_entries = [[float(idx) for idx in entry.split(' ')] for entry in bbox_entry] #  string rbbox entries to float
        bbox_entries = np.array(bbox_entries)
        bbox_entries = xywh2xyxy(bbox_entries)
        bboxes.append(bbox_entries)
    return bboxes


def create_detection_kpts_entries(images_bboxes, images_polygons, images_sizes, images_class_ids):

    """
    Description: one *.txt file per image,  one row per object, row format: class polygon vertices (x0, y0.....xn,yn)
    normalized coordinates [0 to 1].
    zero-indexed class numbers - start from 0


    :param images_paths: list of dataset image filenames
    :param images_polygons: list of per image polygons arrays
    :param images_class_ids:  list of per image class_ids
    :param output_dir: output dir of labels text files
    :return:
    """

    # detection_entries = create_detection_entries(images_bboxes, images_sizes, images_class_ids)
    entries=[]
    for image_polygons, images_size, class_ids, image_detection_entries in zip(images_polygons, images_sizes,
                                                              images_class_ids, images_bboxes):
        image_detection_entries=np.array(image_detection_entries)
        image_polygons=np.array(image_polygons)

        im_height = images_size[0]
        im_width = images_size[1]

        img_kpts = (image_polygons/np.array([im_width, im_height]))
        # concat valid field:
        img_kpts_valid = np.full( [img_kpts.shape[0], img_kpts.shape[1], 1], 2.) # shape: [nobj, nkpts, 1]
        img_kpts = np.concatenate([img_kpts, img_kpts_valid], axis=-1).reshape(img_kpts.shape[0], -1) # flatten kpts per object

        img_entries=[]
        for detection_entry, kpts   in zip(image_detection_entries, img_kpts):
            entry = f"{detection_entry} {' '.join(str(kpt) for kpt in list(kpts.reshape(-1)))}"
            img_entries.append(entry)
        entries.append(img_entries)
    return entries


def normalize_bboxes(images_bboxes, images_sizes, images_class_ids):
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
    for bboxes, images_size, class_ids, in zip(images_bboxes, images_sizes,
                                                                 images_class_ids): # images loop
        im_height = images_size[0]
        im_width = images_size[1]

        # head, filename = os.path.split(image_path)
        bboxes = np.array(bboxes, dtype=float)
        img_bboxes = []

        for bbox, category_id in zip(bboxes, class_ids): # labels in image loop
                # normalize scale:
            xywh_bbox = [bbox[0] / im_width, bbox[1] / im_height,
                             bbox[2] / im_width, bbox[3] / im_height]

            entry = f"{category_id} {' '.join(str(e) for e in xywh_bbox)}"
            img_bboxes.append(entry)
        all_bboxes.append(img_bboxes)
    return all_bboxes

def entries_to_files(batch_entries, out_fnames, output_dir):
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
    print(f'create_per_image_labels_files. labels_output_dir: {output_dir}')
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # catch exception - directory already exists
        pass
    for img_entries, out_fname in zip(batch_entries, out_fnames):
        out_path = f"{output_dir}/{out_fname}"
        with open(out_path, 'w') as f:
            for entry in img_entries:
                f.write(f'{entry}\n')


def dota_entries_to_files(batch_entries, category_names, out_fnames, output_dir):
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
    print(f'create_per_image_labels_files. labels_output_dir: {output_dir}')
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # catch exception - directory already exists
        pass
    for img_entries, out_fname in zip(batch_entries, out_fnames):
            for entry  in zip(img_entries):
                pass
                # arranged_entry =
                # for rbbox in entry[0][:-1]: # skip class - the last list entry
                #     f.write(str(elem)+' ')
                # f.write(str(category_names[int(entry[0][-1])]) + ' ')
                # difficulty = 0
                # f.write(str(difficulty))

    print(f'create_per_image_labels_files. labels_output_dir: {output_dir}')
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # catch exception - directory already exists
        pass
    for img_entries, out_fname in zip(batch_entries, out_fnames):
        out_path = f"{output_dir}/{out_fname}"
        with open(out_path, 'w') as f:
            for entry in img_entries:
                f.write(f'{entry}\n')



# Create a coco like format label file. format:
# a single json file with 4 tables:
#     "info":
#     "licenses":
#     "images": images_records,
#     "categories": categories_records,
#     "annotations": annotatons_records
def create_coco_json_lable_files(images_paths, images_sizes, images_bboxes, images_class_ids,
                   category_names,  category_ids, annotations_output_path):
    """
     :param images_paths: list of dataset image filenames
    :param images_sizes: list of per image [im.height, im.width] tuples
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_class_ids: list of per image class_ids arrays
    :param category_names: list of all dataset's category names
    :param category_ids:  list of all dataset's category ids

    :param annotations_output_path: path for output file storage
    :return:
    """

    print('create_coco_labels_file')

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    # map_categories_id = {}

    # fill category
    for category_name, category_id in zip(category_names, category_ids):

        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": category_names,
                "id": category_id,
                "name": category_name,
            })
            added_category_names.append(category_name)

    images_records = []
    annotatons_records = []
    for example_id, (image_path, image_size, bboxes, categories_list) in enumerate(
            zip(images_paths, images_sizes, images_bboxes, images_class_ids)):

        # images records:

        images_records.append({
            "license": '',
            "file_name": image_path,
            "coco_url": "",
            'width': image_size[1],
            'height': image_size[0],
            "date_captured": str(datetime.now()),
            "flickr_url": "",
            "id": example_id
        })

        # annotatons_records
        for bbox, category_id in zip(bboxes, categories_list):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": category_id,
                "id": anno_id
            }
            )
            anno_id += 1
    date_today = date.today()
    info = {
        "description": " Dataset",
        "url": '',
        # "version": config.get('version', 1.0),
        "year": date_today.year,
        "contributor": '',
        "date_created": str(date_today),
        "licenses": '',
        "categories": categories_records
    }
    output_records = {
        "info": info,
        "licenses": [],
        "images": images_records,
        "categories": categories_records,
        "annotations": annotatons_records
    }
    print(f'Save annotation  in {annotations_output_path}')
    with open(annotations_output_path, 'w') as fp:
        json.dump(output_records, fp)



def write_images_to_file(images, images_out_dir, images_filenames):
            images_sizes=[images[idx].size for idx in np.arange(len(images))]

            print(f'\nimages_out_dir {images_out_dir}')

            # images_filenames=[]
            for idx, (image,image_filename) in enumerate(zip(images, images_filenames)):
                # image_filename = f'img_{idx:06d}.jpg'
                file_path = f'{images_out_dir}/{image_filename}'
                image.save(file_path)
                # images_filenames.append(image_filename)
                print(f'writing image file to disk: {image_filename}')
            return images_filenames
