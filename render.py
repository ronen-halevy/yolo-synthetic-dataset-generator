import yaml
import os
from pathlib import Path

from src.render.draw_dataset_examples import draw_detection_dataset_example, draw_detection_single_file_dataset_example, \
    draw_segmentation_dataset_example, draw_coco_detection_dataset_example


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


def render(nexamples, labels_file_format, image_dir, labels_dir, output_dir, category_names_table, split):
    for idx in range(nexamples):

        if labels_file_format == 'detection_coco_json_format':
            annotations_path = labels_dir
            draw_coco_detection_dataset_example(annotations_path, category_names_table, f'{output_dir}/coco')
        if labels_file_format == 'detection_yolov5':
            draw_detection_dataset_example(image_dir, labels_dir, category_names_table, f'{output_dir}/det1')

        if labels_file_format == 'detection_unified_textfile':
            draw_detection_single_file_dataset_example(labels_dir, image_dir, category_names_table,
                                                       f'{output_dir}/det2')
        if labels_file_format == 'segmentation_yolov5':
            draw_segmentation_dataset_example(image_dir, labels_dir, category_names_table, f'{output_dir}/seg')


if __name__ == "__main__":
    config_file = './config/dataset_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    split = 'train'  # can be 'train', 'test', 'valid'
    image_dir = config['image_dir']  # './dataset/images/{split}/'
    images_out_dir = f'{image_dir.replace("{split}", split)}'
    labels_dir = config['labels_dir'].replace("{split}", split)

    coco_json_labels_file_pth = config['coco_json_labels_file_pth']
    labels_out_dir = coco_json_labels_file_pth.replace('{split}', split)

    labels_out_dir = config['coco_json_labels_file_pth']
    labels_out_dir = labels_out_dir.replace('{split}', split)

    labels_dir = config['labels_dir']

    labels_out_dir = config['labels_all_entries_file']
    labels_out_dir = labels_out_dir.replace("{split}", split)
    nexamples = config['nrender_examples'] + 1

    labels_file_format = config.get('labels_file_format')

    # render =
    # render(image_dir, )
