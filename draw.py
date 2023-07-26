import yaml

from test_dataset.draw_dataset_examples import draw_detection_dataset_example, draw_detection_single_file_dataset_example, draw_segmentation_dataset_example
from test_dataset.utils import increment_path


def main():
    config_file = './test_dataset/test_config.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    output_dir = config['output_dir']
    output_dir=increment_path(output_dir)
    category_names_file = config['category_names_file']
    with open(category_names_file) as f:
        category_names_table = f.readlines()

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
