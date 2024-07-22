#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_yolov_anchors.py
#   Author      : ronen halevy 
#   Created date:  5/3/22
#   Description :
#
# ================================================================
# TODO - set in descending order!
import tensorflow as tf
import yaml

import numpy as np
from sklearn.cluster import KMeans
import argparse
import pathlib
from pathlib import Path
import glob

import os
import matplotlib.pyplot as plt
import math


def plot_scatter_graph(w_h, kmeans):
    """
    Plot kmeans ponts
    """
    plt.scatter(w_h[..., 0], w_h[..., 1], c=kmeans.labels_, alpha=0.5)
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title('K-means Clustering of Boxes Widths and Heights')
    plt.figure(figsize=(10, 10))
    plt.show()


def sort_anchors(anchors):
    anchors_sorted = anchors[(anchors[:, 0] * anchors[:, 1]).argsort()]
    return anchors_sorted


def creat_yolo_anchors(w_h, n_clusters):
    """
    Create anchors by sklearn  KMeans
    :param w_h:
    :type w_h:
    :param n_clusters:
    :type n_clusters:
    :return:
    :rtype:
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')  # Construct with num of clusters (in yolo: nl*na)

    # If data size is less than bcluster - duplicate entries and, add small increments to added entries to produce slitly different clusters:
    if w_h.shape[0] < n_clusters:
        nentries = w_h.shape[0]
        w_h = np.tile(w_h, [math.ceil((n_clusters / w_h.shape[0])), 1])  # dup rows
        random_inc = np.random.rand(w_h.shape[0] - nentries, 2) * np.min(w_h,
                                                                         axis=0) / 10000  # inc neglected: frac of min_val/10000
        w_h[nentries:, :] = w_h[nentries:, :] + random_inc
    kmeans.fit(w_h)

    anchors = kmeans.cluster_centers_  # coordinates of cluster' centers
    sorted_anchors = sort_anchors(anchors).astype(np.float32)
    return sorted_anchors


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return boxes  # cls, xywh


def _list_label_files(path, ext_list):
    """
    Return a list of all files with ext_list extension located at path
    :param path: location of input files, string.
    :param ext_list: dile extension to search for, str
    :return: list of file names found
    """
    try:
        f = []  # label files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
            else:
                raise FileNotFoundError(f'{p} does not exist')
        lb_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ext_list)
        assert lb_files, f'{"prefix"}No label files found'
        return lb_files
    except Exception as e:
        raise Exception(f'Error loading data from {path}: {e}') from e


def read_bboxes_from_label_file(fname, labels_file_format):
    """
    Reads segments label file, retrun class and bbox.
    Input File format-a row per object structured: class, sx1,sy1....sxn,syn
    :param fname: labels file name, str. Fi
    :return:
    lb:  tensor of class,bbox. shape: [nt,5], tf.float32, where nt - num of objects rows in file
    """
    import sys
    from termcolor import cprint
    with open(fname) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if 'segment' in labels_file_format:  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            # img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0], 1])
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            # lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
            #                     1)  # (cls, xywh)
            bboxes = np.array(segments2boxes(segments))
            xy_c = (bboxes[:, :2] + bboxes[:, 2:4]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2])
            bboxes = np.concatenate([xy_c, wh], axis=1)

        elif 'kpts' in labels_file_format or 'keypoint' in labels_file_format or 'detect' in labels_file_format:
            bboxes = np.array(lb)[:, 1:5]
        else:
            print(f'{labels_file_format} labels file format not supported. Terminating')
            exit(1)

        bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes


def save_to_file(anchors_out_file, anchors):
    base_dir, fname = os.path.split(anchors_out_file)
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    data = {'anchors': anchors.tolist()}
    with open(anchors_out_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='create_anchors_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        config_file = yaml.safe_load(stream)

    n_layers = config_file['n_layers']
    n_anchors = config_file['n_anchors']
    n_clusters = n_layers * n_anchors
    anchors_out_file = config_file['anchors_out_file']
    labels_file_format = config_file['labels_file_format']

    ext_list = 'txt'
    path = config_file['labels_dir']
    label_files = _list_label_files(path, ext_list)

    # bboxes=np.zeros([0,11])
    bboxes = np.zeros([0, 4])
    print(
        f'Creating \033[1m{n_clusters} ({n_layers} layers * {n_anchors} anchors)\033[0m based on \033[1m{labels_file_format}\033[0m format labels')

    for idx, label_file in enumerate(label_files):
        # extract class, bbox and segment from label file:
        bboxes_i = read_bboxes_from_label_file(label_file,
                                               labels_file_format)  # list[ni] of image boxes, bboxes_i shape:[4]
        bboxes = np.concatenate([bboxes, bboxes_i], axis=0)  # labels shape: [b*nt,5] where b nof label files

    # wh = bboxes[:,3:5]#- labels[:,1:3]
    wh = bboxes[:, 2:4]  # - labels[:,1:3]

    anchors = creat_yolo_anchors(wh, n_clusters)
    im_size = config_file['yolo_image_size']
    anchors = np.array(anchors * im_size).reshape([n_layers, n_anchors, 2]).tolist()

    with open(anchors_out_file, 'w') as file:
        yaml.dump(anchors, file)

    # save_to_file(anchors_out_file, anchors * im_size)
    print(f'result anchors:\n{anchors}')
    print(f'anchors_out_file: {anchors_out_file}')


if __name__ == '__main__':
    main()
