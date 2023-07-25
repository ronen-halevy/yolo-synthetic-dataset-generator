import numpy as np
import os

# create a row labels text file per image. format:
# x0l,y0l,x0h,y0h,c, ......
# .
# xnl,ynl,xnh,ynh,c

def segmentation_labels_formatter(images_paths, images_polygons, images_sizes, images_objects_categories_indices,

                                  output_dir):
    """

    :param images_paths: list of dataset image filenames
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_sizes:
    :param images_objects_categories_indices:  list of per image categories_indices arrays
    :param output_dir: output dir of labels text files
    :return:
    """
    print('segmentation_labels_formatter')
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        # directory already exists
        pass
    for image_polygons, image_path, images_size, categories_indices in zip(images_polygons, images_paths, images_sizes,
                                                              images_objects_categories_indices):
        im_height = images_size[0]
        im_width = images_size[1]

        head, filename = os.path.split(image_path)
        labels_filename = f"{output_dir}/{filename.rsplit('.', 1)[0]}.txt"
        image_polygons=np.array(image_polygons)/images_size
        with open(labels_filename, 'w') as f:
            for image_polygon, category_id in zip(image_polygons, categories_indices):
                entry = f"{category_id} {' '.join(str(vertix) for vertix in list(image_polygon.reshape(-1)))}\n"


                f.write(entry)

