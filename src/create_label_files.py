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


# Create a coco like format label file. format:
# a single json file with 4 tables:
#     "info":
#     "licenses":
#     "images": images_records,
#     "categories": categories_records,
#     "annotations": annotatons_records


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
