import yaml
import os
from pathlib import Path

from test_dataset.draw_dataset_examples import draw_detection_dataset_example, draw_detection_single_file_dataset_example, draw_segmentation_dataset_example,draw_coco_detection_dataset_example

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def main():
    config_file = './test_dataset/test_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config['output_dir']
    output_dir=increment_path(output_dir)
    category_names_file = config['category_names_file']
    with open(category_names_file) as f:
        category_names_table = f.readlines()

    if 'coco_detection_datase' in config['label_file_formats'].keys():
        draw_coco_detection_dataset_example(config['label_file_formats']['coco_detection_datase']['annotations_path'], category_names_table, output_dir)

    if 'yolov5_detection_format' in config['label_file_formats'].keys():
        image_dir = config['label_file_formats']['yolov5_detection_format']['image_dir']
        label_dir = config['label_file_formats']['yolov5_detection_format']['label_dir']
        draw_detection_dataset_example(image_dir, label_dir, category_names_table, output_dir)

    if 'single_label_file_format' in config['label_file_formats'].keys():
        label_path=config['label_file_formats']['single_label_file_format']['label_path']
        draw_detection_single_file_dataset_example(label_path, category_names_table, output_dir)

    if 'yolov5_segmentation_format' in config['label_file_formats'].keys():
        image_dir = config['label_file_formats']['yolov5_segmentation_format']['image_dir']
        label_dir = config['label_file_formats']['yolov5_segmentation_format']['label_dir']
        draw_segmentation_dataset_example(image_dir, label_dir, category_names_table, output_dir)


if __name__ == "__main__":
    main()
