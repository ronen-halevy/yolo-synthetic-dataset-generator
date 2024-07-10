import yaml
import os
from pathlib import Path

from src.render.render_utils import draw_detection_dataset_example, draw_detection_single_file_dataset_example, \
    draw_segmentation_dataset_example, draw_coco_detection_dataset_example, draw_obb_dataset_example

import random

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    """
    :param path:
    :type path:
    :param exist_ok:
    :type exist_ok:
    :param sep:
    :type sep:
    :param mkdir:
    :type mkdir:
    :return:
    :rtype:
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def render(nexamples, labels_file_format, image_dir, labels_dir, output_dir, category_names_table, split):
    listdir = [filename for filename in os.listdir(image_dir) if
               filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]


    sel_files=random.sample(listdir,  k=min(nexamples, len(listdir)))

    for idx, sel_fname in enumerate(sel_files):
        image_path = f'{image_dir}/{sel_fname}'
        label_path = f'{labels_dir}/{Path(sel_fname).stem}.txt'
        if not os.path.isfile(label_path):
            print(f'Label file {label_path} not found. terminating!')
            exit(1)

        if labels_file_format == 'detection_coco_json_format':
            annotations_path = labels_dir
            draw_coco_detection_dataset_example(annotations_path, category_names_table, f'{output_dir}/coco')
        elif labels_file_format == 'detection_yolov5':
            draw_detection_dataset_example(image_path, label_path, category_names_table, f'{output_dir}/det1')
        elif labels_file_format == 'detection_unified_textfile':
            draw_detection_single_file_dataset_example(image_path, label_path, category_names_table,
                                                       f'{output_dir}/det2')
        elif labels_file_format == 'segmentation_yolov5':
            draw_segmentation_dataset_example(image_path, label_path, category_names_table, f'{output_dir}/seg')
        elif labels_file_format == 'obb':
            draw_obb_dataset_example(image_path, label_path, category_names_table, f'{output_dir}/det1')
        else:
            print(f'Unknow labels_file_format: {labels_file_format}')

if __name__ == "__main__":
    config_file = './config/dataset_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    split = config['split_to_render'] # 'train'  # can be 'train', 'test', 'valid'
    labels_file_format = config.get('labels_file_format')

    if labels_file_format in ['segmentation_yolov5', 'detection_yolov5', 'kpts_detection_yolov5', 'obb']:
        labels_dir = f'{config["output_dir"]}/{config["labels_dir"]}/{split}'
        images_dir = f'{config["output_dir"]}/{config["image_dir"]}/{split}'
    elif labels_file_format == 'detection_coco_json_format':
        coco_json_labels_file_path = config['coco_json_labels_file_path']
        labels_dir = coco_json_labels_file_path.replace('{split}', split)
        images_dir=None # complete path within json file
    elif labels_file_format == 'detection_unified_textfile':
        labels_dir = config['labels_all_entries_file'].replace("{split}", split)
        images_dir = config["image_dir"].replace("{split}", split) # an offset for image filenames located in labels
    else:
        raise('Unknown or missing labels_file_format! configuration')
    nexamples = config['nrender_examples'] + 1
    labels_file_format = config.get('labels_file_format')

    class_names_file=config['category_names_file']

    output_dir = config['test_output_dir']
    print('\nrendering dataset images with bbox and mask overlays\n')
    output_dir=increment_path(output_dir)


    category_names = [c.strip() for c in open(class_names_file).readlines()]
    render(nexamples, labels_file_format, images_dir, labels_dir, f'{output_dir}/{split}', category_names, split)

