import numpy as np
import os

# create a row labels text file per image. format:
# x0l,y0l,x0h,y0h,c, ......
# .
# xnl,ynl,xnh,ynh,c

def raw_text_files_labels_formatter(images_paths, images_bboxes, images_sizes, images_objects_categories_indices,

                                  output_dir):
    """

    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_sizes:
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    print('create_per_image_labels_files')
    output_dir = f'{output_dir}labels/'
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # directory already exists
        pass
    for bboxes, image_path, images_size, categories_indices in zip(images_bboxes, images_paths, images_sizes,
                                                              images_objects_categories_indices):
        im_height = images_size[0]
        im_width = images_size[1]

        head, filename = os.path.split(image_path)
        labels_filename = f"{output_dir}{filename.rsplit('.', 1)[0]}.txt"
        with open(labels_filename, 'w') as f:
            for bbox, category_id in zip(bboxes, categories_indices):
                bbox_arr = np.array(bbox, dtype=float)
                # [xmin, ymin, w,h] to [x_c, y_c, w, h]
                xcycwh_bbox = [(bbox_arr[0] + bbox_arr[2] / 2) , (bbox_arr[1] + bbox_arr[3] / 2) ,
                               bbox_arr[2], bbox_arr[3] ]
                # normalize size:
                xcycwh_bbox = [xcycwh_bbox[0] / im_width,xcycwh_bbox[1] / im_height,
                               xcycwh_bbox[2] / im_width, xcycwh_bbox[3] / im_height]
                entry = f"{category_id} {' '.join(str(e) for e in xcycwh_bbox)}\n"
                f.write(entry)

