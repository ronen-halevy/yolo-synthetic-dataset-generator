import json
from datetime import date, datetime
import numpy as np
import os
import yaml

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

def create_keypoints_label_files(images_polygons, images_sizes, images_class_ids,labels_fnames,
                                  output_dir):

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
    print(f'create_keypoints_label_files. output_dir: {output_dir}')
    # create out dirs if needed - tbd never needed...
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # catch if directory already exists
        pass
    ## create label files

    for image_polygons, labels_fname, images_size, class_ids in zip(images_polygons, labels_fnames, images_sizes,
                                                              images_class_ids):
        polygons=np.array(image_polygons)
        bbpxes = segments2bboxes_batch(polygons, width=640, height=640)
        im_height = images_size[0]
        im_width = images_size[1]
        # normalize:
        bbpxes/=[im_width, im_height, im_width, im_height]
        kpts = (polygons/np.array([im_width, im_height]))
        # concat valid field:
        kpts_valid = np.full( [kpts.shape[0], kpts.shape[1], 1], 2.) # shape: [nobj, nkpts, 1]
        kpts = np.concatenate([kpts, kpts_valid], axis=-1).reshape(kpts.shape[0], -1) # flatten kpts per object
        entries = np.concatenate([bbpxes, kpts], axis=1)


        labels_filename = f"{output_dir}/{labels_fname}"
        print(f'labels_filename: {labels_filename}')

        # normalize sizes:
        # image_polygons=[image_polygon/np.array(images_size) for image_polygon in image_polygons]
        with open(labels_filename, 'w') as f:
            category_id=0 # assumed a single class in kpts mode
            for entry  in entries:
                entry = f"{category_id} {' '.join(str(vertix) for vertix in list(entry.reshape(-1)))}\n"
                f.write(entry) # fill label file with entrie

def create_segmentation_label_files(images_polygons, images_sizes, images_class_ids,labels_fnames,
                                  output_dir):
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
    print(f'create_segmentation_label_files. output_dir: {output_dir}')
    # create out dirs if needed - tbd never needed...
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # catch if directory already exists
        pass
    ## create label files
    for image_polygons, labels_fname, images_size, class_ids in zip(images_polygons, labels_fnames, images_sizes,
                                                              images_class_ids):
        im_height = images_size[0]
        im_width = images_size[1]
        labels_filename = f"{output_dir}/{labels_fname}"
        print(f'labels_filename: {labels_filename}')

        # normalize sizes:
        image_polygons=[image_polygon/np.array(images_size) for image_polygon in image_polygons]
        with open(labels_filename, 'w') as f:
            for image_polygon, category_id in zip(image_polygons, class_ids):
                entry = f"{category_id} {' '.join(str(vertix) for vertix in list(image_polygon.reshape(-1)))}\n"
                f.write(entry) # fill label file with entries


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


def create_detection_entries(images_bboxes, images_sizes, images_class_ids):
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

    entries = []
    for bboxes, images_size, class_ids, in zip(images_bboxes, images_sizes,
                                                                 images_class_ids): # images loop
        im_height = images_size[0]
        im_width = images_size[1]

        # head, filename = os.path.split(image_path)
        bboxes = np.array(bboxes, dtype=float)
        img_entries = []

        for bbox, category_id in zip(bboxes, class_ids): # labels in image loop
                # normalize scale:
            xywh_bbox = [bbox[0] / im_width, bbox[1] / im_height,
                             bbox[2] / im_width, bbox[3] / im_height]

            entry = f"{category_id} {' '.join(str(e) for e in xywh_bbox)}\n"
            img_entries.append(entry)
        entries.append(img_entries)
    return entries

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
                f.write(entry)


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
