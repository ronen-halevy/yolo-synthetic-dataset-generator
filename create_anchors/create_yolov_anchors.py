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
    kmeans = KMeans(n_clusters=n_clusters)  # Construct with num of clusters (in yolo - 9 (3*3))
    kmeans.fit(w_h)
    anchors = kmeans.cluster_centers_  # coordinates of cluster' centers
    sorted_anchors = sort_anchors(anchors).astype(np.float32)
    # plot_scatter_graph(w_h, kmeans)
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


def read_label_from_file(fname):
    """
    Reads segments label file, retrun class and bbox.
    Input File format-a row per object structured: class, sx1,sy1....sxn,syn
    :param fname: labels file name, str. Fi
    :return:
    lb:  tensor of class,bbox. shape: [nt,5], tf.float32, where nt - num of objects rows in file
    """
    with open(fname) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if any(len(x) > 6 for x in lb):  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            # img_index = tf.tile(tf.expand_dims(tf.constant([idx]), axis=-1), [classes.shape[0], 1])
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
                                1)  # (cls, xywh)
        lb = np.array(lb, dtype=np.float32)
    return lb

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

    n_clusters = config_file['n_clusters']
    anchors_out_file = config_file['anchors_out_file']

    ext_list = 'txt'
    path = config_file['labels_dir']
    label_files = _list_label_files(path, ext_list)

    labels=np.zeros([0,5])
    for idx, label_file in enumerate(label_files):
        # extract class, bbox and segment from label file:
        label = read_label_from_file(label_file)
        labels=np.concatenate([labels, label], axis=0) # labels shape: [b*nt,5] where b nof label files

    wh = labels[:,3:5]- labels[:,1:3]
    anchors = creat_yolo_anchors(wh, n_clusters)
    im_size = 640
    save_to_file(anchors_out_file, anchors * im_size)
    print(f'result anchors:\n{anchors}')
    print(f'anchors_out_file: {anchors_out_file}')


if __name__ == '__main__':
    main()

