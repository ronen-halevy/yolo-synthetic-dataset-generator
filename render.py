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

def main():
    config_file = './config/dataset_config.yaml'

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config['test_output_dir']
    print('\nrendering dataset images with bbox and mask overlays\n')
    output_dir=increment_path(output_dir)
    category_names_file = config['category_names_file']
    with open(category_names_file) as f:
        category_names_table = f.read().splitlines()
    splits=config['splits']
    # loop on configured splits e.g. [train, val, test]. replace various paths strings accordingly:
    for idx in range(config['nrender_examples']+1):
        for split in splits.keys():
            if splits[split] > 0:
                if 'coco_detection_dataset_labels_path' in config:
                    annotations_path = config['coco_detection_dataset_labels_path'].replace("{split}", split)
                    draw_coco_detection_dataset_example(annotations_path, category_names_table, f'{output_dir}/{split}/coco')

                if 'detection_label_text_files_path' in config:
                    image_dir = config['image_dir'].replace('{split}', 'train')
                    label_dir = config['detection_label_text_files_path'].replace('{split}',  split)
                    draw_detection_dataset_example(image_dir, label_dir, category_names_table, f'{output_dir}/{split}/det1')

                if 'detection_label_unified_file_path' in config:
                    label_path=config['detection_label_unified_file_path'].replace('{split}', split)
                    draw_detection_single_file_dataset_example(label_path, category_names_table, f'{output_dir}/{split}/det2')

                if 'segmentation_label_files_path' in config:
                    image_dir = config['image_dir'].replace('{split}',  split)
                    label_dir = config['segmentation_label_files_path'].replace('{split}',  split)
                    draw_segmentation_dataset_example(image_dir, label_dir, category_names_table, f'{output_dir}/{split}/seg')


if __name__ == "__main__":
    main()
