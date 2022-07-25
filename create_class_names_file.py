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

config_file = 'config/config.yaml'
shapes_in_file = 'config/shapes.yaml'
class_out_files = 'dataset/class.names'

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

with open(shapes_in_file, 'r') as stream:
    shapes = yaml.safe_load(stream)

class_mode = config['class_mode']
labels = []
with open(class_out_files, 'w') as f:
    for shape in shapes:
        label = shape[
            'color'] if class_mode == 'color' else f"{shape['color']}_{shape['shape']}" if class_mode == 'color_and_shape' else \
            shape[
                'shape']
        if label not in labels:
            labels.append(label)
            f.write("%s\n" % label)
