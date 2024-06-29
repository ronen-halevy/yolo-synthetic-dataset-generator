import yaml
import os
from pathlib import Path
from src.utils import increment_path


from src.render.draw_dataset_examples import draw_detection_dataset_example, draw_detection_single_file_dataset_example, \
    draw_segmentation_dataset_example, draw_coco_detection_dataset_example



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
    split = config['split_to_render'] # 'train'  # can be 'train', 'test', 'valid'
    labels_file_format = config.get('labels_file_format')

    if labels_file_format in ['segmentation_yolov5', 'detection_yolov5']:
        labels_dir = f'{config["labels_dir"]}/{split}'
        images_dir = f'{config["image_dir"]}/{split}'

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

