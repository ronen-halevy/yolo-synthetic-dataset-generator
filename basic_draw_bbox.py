import yaml

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import os

def draw_bounding_box(image, boxes, thickness=1):
    colors = list(ImageColor.colormap.values())
    color = colors[0]
    thickness = 1
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, w, h = box
        # xmin = xc-w/2
        # ymin = yc-h/2
        print((xmin, ymin), (xmin, ymin+h), (xmin+w, ymin+h), (xmin+w, ymin))
        draw.line([(xmin, ymin), (xmin, ymin+h), (xmin+w, ymin+h), (xmin+w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image




label_file_format = 'yolov5_tf_format' # ['yolov5_pytorch_format', 'yolov5_tf_format']

# if label_file_format == 'coco_yolov3_format':
#     image_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/img_000001.jpg'
#     lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/labels/img_000001.txt'
#


if label_file_format == 'yolov5_pytorch_format':

    image_path = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/images/img_000001.jpg'
    lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/labels/img_000001.txt'
    if os.path.isfile(lb_file):
        nf = 1  # label found
        with open(lb_file) as f:
            bboxes = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    bboxes = np.array(bboxes, dtype=float)[:, 1:5] * [image.width, image.height, image.width, image.height]
    #convert to x_Center, y_center to cmin, ymin
    bboxes[:,0]= bboxes[:,0]-bboxes[:,2]/2
    bboxes[:,1]= bboxes[:,1]-bboxes[:,3]/2
elif label_file_format == 'yolov5_tf_format':
    lb_file = '/home/ronen/devel/PycharmProjects/shapes-dataset/dataset/train/all_entries.txt'
    with open(lb_file, 'r') as f:
        annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        example = annotations[0].split()
        image_path = example[0]
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:,0:4]
        # from xmin ymin xmax yamx to xmin yamin, w,h

        bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
        bboxes[:,3] = bboxes[:,3] - bboxes[:,1]


annotated_bbox_image = draw_bounding_box(image, bboxes)

figure(figsize=(10, 10))
plt.imshow(annotated_bbox_image)
plt.show()
