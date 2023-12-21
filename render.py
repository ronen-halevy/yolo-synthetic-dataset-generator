import yaml
import os
from pathlib import Path

from src.render.draw_dataset_examples import draw_detection_dataset_example, draw_detection_single_file_dataset_example, draw_segmentation_dataset_example,draw_coco_detection_dataset_example

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
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

def render(image_dir, labels_path, output_dir, category_names_table):
    config_file = './config/dataset_config.yaml'

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)


    # category_names_file = config['category_names_file']
    # with open(category_names_file) as f:
    #     category_names_table = f.read().splitlines()
    # splits=config['splits']
    # loop on configured splits e.g. [train, val, test]. replace various paths strings accordingly:
    for idx in range(config['nrender_examples']+1):
        # for split in splits.keys():
        #     if splits[split] > 0:
                if config.get('labels_file_format') == 'detection_coco_json_format':
                    annotations_path = labels_path
                    draw_coco_detection_dataset_example(annotations_path, category_names_table, f'{output_dir}/coco')
                if config.get('labels_file_format') == 'detection_yolov5':
                    image_dir = image_dir
                    label_dir = labels_path
                    draw_detection_dataset_example(image_dir, label_dir, category_names_table, f'{output_dir}/det1')

                if config.get('labels_file_format') == 'detection_unified_textfile':
                    label_path=labels_path
                    draw_detection_single_file_dataset_example(label_path, category_names_table, f'{output_dir}/det2')
                if config.get('labels_file_format') == 'segmentation_yolov5':
                    image_dir = image_dir
                    label_dir = labels_path
                    draw_segmentation_dataset_example(image_dir, label_dir, category_names_table, f'{output_dir}/seg')


if __name__ == "__main__":
    main()
