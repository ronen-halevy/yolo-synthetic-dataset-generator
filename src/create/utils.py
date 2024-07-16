import numpy as np
import os
from PIL import Image, ImageColor, ImageDraw
from shapely.geometry import Polygon


def calc_iou(polygon1, polygons):
    """
    Calc iou between polygon1 and polygons - a list of polygons
    :param polygon: polygon vertices, np.array, shape: [nvertices,2]
    :param polygons: a list of np,array polygons of shape [nvertices,2]
    :return: a list, iou of polygon1 and polygons
    """
    polygon1 = Polygon(polygon1)
    iou = []
    for polygon2 in polygons:
        polygon2 = Polygon(polygon2)
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou.append(intersect / union)
    return np.array(iou)
def write_entries_to_files(batch_entries, out_fnames, output_dir):
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
            for idx, (image,image_filename) in enumerate(zip(images, images_filenames)):
                file_path = f'{images_out_dir}/{image_filename}'
                image.save(file_path)
            return images_filenames


def draw_images(images_polygons, images_objects_colors=None, images_size=None, bg_color_set=['red']):
    # related label file has same name with .txt ext - split filename, replace ext to txt:
    images = []
    for idx, (image_polygons, image_objects_color, image_size) in enumerate(
            zip(images_polygons, images_objects_colors, images_size)):

        # save image files
        bg_color = np.random.choice(bg_color_set)
        image = Image.new('RGB', tuple(image_size), bg_color)
        draw = ImageDraw.Draw(image)

        for image_polygon, image_object_color in zip(image_polygons, image_objects_color):
            color = np.random.choice(image_object_color)
            draw.polygon(image_polygon.flatten().tolist(), fill=ImageColor.getrgb(color))
        images.append(image)
    return images