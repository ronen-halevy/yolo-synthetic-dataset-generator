import json
from datetime import date, datetime
import numpy as np
import os


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




