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
import json

shapes_in_file = 'shapes.json'
class_out_files = 'class.names'

with open(shapes_in_file) as f:
    shapes = json.load(f)['shapes']
classes=[]
with open(class_out_files, 'w') as f:
    for shape in shapes:
        if shape['label'] not in classes:
            classes.append(shape['label'] )
            f.write("%s\n" % shape['label'] )
