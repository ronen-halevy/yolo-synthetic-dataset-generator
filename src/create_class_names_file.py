#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_class_names_file.py
#   Author      : ronen halevy 
#   Created date:  4/24/22
#   Description : Create a class names file for dataset usage.
#
# ================================================================
import yaml



def create_class_mames(config_file='config/config.yaml', shapes_in_file='config/shapes.yaml',
                       class_names_out_file='dataset/class.names'):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    with open(shapes_in_file, 'r') as stream:
        shapes = yaml.safe_load(stream)

    class_mode = config['class_mode']
    labels = []
    print(f'Creating {class_names_out_file}.\n Running...')
    with open(class_names_out_file, 'w') as f:
        for shape in shapes:
            label = shape[
                'color'] if class_mode == 'color' else f"{shape['color']}_{shape['shape']}" if class_mode == 'color_and_shape' else \
                shape[
                    'shape']
            if label not in labels:
                labels.append(label)
                f.write("%s\n" % label)
    print(f'Completed!')

if __name__ == '__main__':
    create_class_mames()