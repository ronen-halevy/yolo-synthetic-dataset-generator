from pathlib import Path

import yaml
from PIL import Image
from PIL import ImageDraw
from PIL import Image as im
from PIL import ImageColor
from PIL import ImageFont
import numpy as np
import cv2
import random


def plot_skeleton_kpts(im, img_kpts, steps, skeleton):
    """
    Plot keypoints overlays on image. Plot lkeypoints connecting lines according to skeleton lines list

    :param im: shape: [h,w,c], [0, 255]
    :param kpts: list[nt] each list entry: [nkpts, 3], float, values range: [0, max(h,w)]
    :param steps: steps between kpts. either 3 for x,y,occlusion ir 2 if a kpts is represented by x,y coordinates.
    :param skeleton: a list of kpts pairs for rendering connrcting lines. list size: [nskeleton_lines][2]. skip  if None
    :return: No return
    :rtype:
    """

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if skeleton is None: # as default, use pose kpts skeleton:
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    for kpts in img_kpts:
        num_kpts = len(kpts) // steps

    # draw kpts as circles:
        for kid in range(num_kpts): # loop on kpt id
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    occlusion = kpts[steps * kid + 2]
                    if occlusion < 0.5: # occlusion originally 0,1,2 divided by steps
                        continue
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
            pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
            if steps == 3:
                conf1 = kpts[(sk[0]-1)*steps+2]
                conf2 = kpts[(sk[1]-1)*steps+2]
                if conf1<0.5 or conf2<0.5:
                    continue
            if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
                continue
            if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
                continue
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


####

def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    """
    Description: Draws a text which starts at xmin,ymin bbox corner

    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_margin_factor = 0.05

    left, top, right, bottom = zip(*[font.getbbox(display_str) for display_str in display_str_list])
    text_heights = tuple(map(lambda i, j: i - j, bottom, top))
    text_widths = tuple(map(lambda i, j: i - j, right, left))

    text_margins = np.ceil(text_margin_factor * np.array(text_heights))
    text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

    for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
            zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):
        left, top, right, bottom = font.getbbox(display_str)
        text_height = bottom - top
        text_width = right - left

        text_margin = np.ceil(text_margin_factor * text_height)

        draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                        (xmint + text_width + text_margin, text_bottom)),
                       fill=tuple(color))

        draw.text((xmint + text_margin, text_bottom - text_height - 3 * text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


def plot_one_box(x, im, label=None, color='black', line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)

def draw_bbox_xywh(image, bboxes, category_names, thickness=1):
    # annotated_bbox_image = draw_bounding_box(image, bboxes)
    colors = list(ImageColor.colormap.values())
    color = colors[7]
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        xmin, ymin, w, h = bbox
        draw.line([(xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)


    text_box_color = [255, 255, 255]
    draw_text_on_bounding_box(image, np.array(bboxes)[..., 1],
                                                     np.array(bboxes)[..., 0], text_box_color,
                                                     category_names, font_size=15)

    return image


def draw_bbox_xyxy(image, bboxes, category_names, thickness=1):
    # annotated_bbox_image = draw_bbox_xyxy(image, bboxes)
    colors = list(ImageColor.colormap.values())
    color = colors[7]
    draw = ImageDraw.Draw(image)
    for bbox_xyxy in bboxes:
        draw.line([(bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (bbox_xyxy[4], bbox_xyxy[5]), (bbox_xyxy[6], bbox_xyxy[7]),
                   (bbox_xyxy[0], bbox_xyxy[1])],
                  width=thickness,
                  fill=color)




    text_box_color = [255, 255, 255]
    draw_text_on_bounding_box(image, np.array(bboxes)[..., 1],
                                                     np.array(bboxes)[..., 0], text_box_color,
                                                     category_names, font_size=15)

    return image


def read_detection_dataset_entry(image_path, label_path):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged as a text file per image arrangement.
    Label's file name corresponds to image filename with .txt extension. Label file format matches Ultralytics
    yolov5 detection dataset, i.e.  5 words per object rows, each holds category_id and bbox  x_center, y_center, w, h
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, x_center, y_center, w, h
    :type label_path: str
    :return:
    image: image read from file
    :type: PIL
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    """

    with open(label_path) as f:
        lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_ids = np.array(lables)[:, 0].astype(int)
    bboxes = np.array(lables, dtype=float)[:, 1:5] * [image.width, image.height, image.width, image.height]

    # convert from x_center, y_center to xmin, ymin
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    return image, bboxes, category_ids


def read_single_file_detection_dataset(label_path, image_dir):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged in a single text file, common to all dataset examples, a row per an image example. row's format:
     category_id, x_center, y_center, w, h. his method chooses randomly a file row, which holds an image path, and a
     set of bbox & category id per each object. Method returns the read image, a list of bboxes and ids.

    :param label_path: label file path.
    :type label_path: str
    :return:
    image , bboxes, category ids

    :rtype:

    """
    with open(label_path, 'r') as f:
        annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        example = random.choice(annotations).split()
        image_path = f'{image_dir}/{example[0]}'
        image = Image.open(image_path)
        bboxes = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 0:4]
        category_ids = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])[:, 4].astype(int)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return image, bboxes, category_ids, image_path


def read_segmentation_dataset_entry(image_path, label_path):
    """
    Description:
    This method reads segmentation dataset entry. Following Ultralytics yolo convention,a dataset entry is defined
    by an image file and a label file with same name but latter extention is .txt. Label files formatted with a row per
    object, with a category_id and polygon's coordinates within.
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, polygon's coordinates
    :type label_path: str
    :return:
    image: image read from file. type: PIL
    polygons: list[nt], of nt polygons format: [[[xj,j],j=0:nv_i], i=0:nt], where nv_i nof vertices, nt: nof objects
    bboxes: list[nt]. entry format:  [[xmin,ymin,w,h], i=0:nt]
    category_ids:  a list of per-image-object category id
    """
    image = Image.open(image_path)
    size = np.array([image.height, image.width]).reshape(1, -1)
    with open(label_path) as f:
        entries = [x.split() for x in f.read().strip().splitlines() if len(x)]
    if 'entries' in locals():
        polygons = [np.array(entry)[1:].reshape(-1, 2).astype(float) for entry in entries]
        category_ids = [np.array(entry)[0].astype(int) for entry in entries]
        polygons = [(polygon * size).astype(int) for polygon in polygons]
        bboxes = []
        for polygon, category_id in zip(polygons, category_ids):
            x, y = polygon[:, 0], polygon[:, 1]
            bbox = [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()]
            bboxes.append(bbox)
    else:
        print(f'labels files {label_path} does not exist!')
        bboxes = []
        category_ids = []
        polygons = []

    return image, polygons, bboxes, category_ids


def read_obb_dataset_entry(image_path, label_path):
    """
    Description:
    This method demonstrates the reading and rendering of a detection dataset entry, where the dataset labels are
    arranged as a text file per image arrangement.
    Label's file name corresponds to image filename with .txt extension. Label file format matches Ultralytics
    yolov5 detection dataset, i.e.  5 words per object rows, each holds category_id and bbox  x_center, y_center, w, h
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, x_center, y_center, w, h
    :type label_path: str
    :return:
    image: image read from file
    :type: PIL
    bboxes: a list of per-image-object bboxes. format:  xmin, ymin, w, h
    category_ids:  a list of per-image-object category id
    """

    with open(label_path) as f:
        lables = [x.split() for x in f.read().strip().splitlines() if len(x)]
    image = Image.open(image_path)
    category_names = np.array(lables)[:, 8]  #
    polygons = np.array(lables)[:, 0:8].astype(np.float32) * [image.width, image.height, image.width, image.height,
                                                              image.width, image.height, image.width, image.height]
    return image, polygons, category_names

def read_kpts_dataset_entry(image_path, label_path):
    """
    Description:
    This method reads segmentation dataset entry. Following Ultralytics yolo convention,a dataset entry is defined
    by an image file and a label file with same name but latter extention is .txt. Label files formatted with a row per
    object, with a category_id and polygon's coordinates within.
    :param image_path: image file path
    :type image_path: str
    :param label_path: label file path. row's format: category_id, polygon's coordinates
    :type label_path: str
    :return:
    image: image read from file. type: PIL
    bboxes: list[nt]. entry format:  [[x,y,x,y], i=0:nt]
    kpts: list[nti][nkpts*step] where nti: nof objects in image, step=2 if entry is x,y and 3 if x,y,occlusion
    category_ids:  a list of per-image-object category id - todo
    """
    image = Image.open(image_path)
    size = np.array([image.height, image.width]).reshape(1, -1)
    with open(label_path) as f:
        # entry len: 4+nkpts*3, where 3=x+y+visibility, where visibility=[0,2]
        entries = [x.split() for x in f.read().strip().splitlines() if len(x)]
    if 'entries' in locals():
        # entry struct: [cls,bbox,bkpts*3], with [x,y,visibilty] fields per a keypoint.
        bboxes = np.array([np.array(entry)[1:5].astype(float) for entry in entries]) # array [nt,4]
        kpts = [np.array(entry)[5:].astype(float) for entry in entries] # list[nt] of arrays[nkpts*3]
        kpts = [np.array(entry)[5:].astype(float) for entry in entries] # list[nt] of arrays[nkpts*3]
        step = 3 # assumed x,y,occlusion
        kpts = [(kpt.reshape([-1,step])* np.array([image.width, image.height, 1])).reshape([-1]).tolist() for kpt in kpts] # array [nt,nkpts*3] # scale to img size


    else:#todo
        print(f'labels files {label_path} does not exist!')
        bboxes = []
        kpts = []

    return image, bboxes, kpts


def draw_detection_dataset_example(image_path, label_path, category_names_table):
    [image, bboxes, category_ids] = read_detection_dataset_entry(image_path, label_path)
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image

def draw_obb_dataset_example(image_path, label_path):
    [image, polygons, category_names] = read_obb_dataset_entry(image_path, label_path)
    draw_bbox_xyxy(image, polygons, category_names)
    return image

def draw_kpts_dataset_example(image_path, label_path):
    [im, bboxes, kpts] = read_kpts_dataset_entry(image_path, label_path)
    image=np.array(im)
    # draw_bbox_xywh(image, bboxes, ['']) # assumed a single category
    bboxes = bboxes * np.tile([im.height, im.width],[2])
    bboxes = np.concatenate([bboxes[:, 0:2] - bboxes[:, 2:4] / 2, bboxes[:, 0:2] + bboxes[:, 2:4] / 2], axis=-1)

    for bbox in bboxes:
        plot_one_box(bbox, image, line_thickness=3)

    steps=3 # each entry is [x,y,occlusiuon]
    skeleton=[]
    plot_skeleton_kpts(image, kpts, steps, skeleton)

    return image

def draw_detection_single_file_dataset_example(label_path, image_dir, category_names_table):
    [image, bboxes, category_ids, image_path] = read_single_file_detection_dataset(label_path, image_dir)
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image


def draw_segmentation_dataset_example(image_path, label_path, category_names_table):
    """
    Draw a randomly selected image with segmentation, bbox and class labels overlays

    :param image_dir: images directory for a random image selection
    :type image_dir: str
    :param label_dir: segmentation labels directory, a label file per an image, with same filename but .txt ext
    :type label_dir: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
    :return:
    :rtype:
    """

    # arrange output elements:
    [image, polygons, bboxes, category_ids] = read_segmentation_dataset_entry(image_path, label_path)
    # fill objects with  masks by polygons:
    array_image = np.array(image)
    # if polygons:
    for polygon, category_id in zip(polygons, category_ids):
        color = np.random.randint(low=0, high=255, size=3).tolist()
        cv2.fillPoly(array_image, np.expand_dims(polygon, 0), color=color)
    image = im.fromarray(array_image)
    ImageDraw.Draw(image)
    # extract category names by ids:
    category_names = [category_names_table[category_id] for category_id in category_ids]
    draw_bbox_xywh(image, bboxes, category_names)
    return image


def draw_coco_detection_dataset_example(annotations_path, category_names_table):
    """
    Draw a randomly selected image with bboxes and class labels overlays according to COCO format label files

    :param annotations_path: coco format annotations json file path
    :type annotations_path: str
    :param category_names_table: list of dataset's category - to annotate image with a label
    :type category_names_table: list of str
    :return:
    :rtype:
    """
    with open(annotations_path) as file:
        annotations = yaml.safe_load(file)

    # randomy select an image index from dataset:
    if len(annotations['images']):
        image_index = np.random.randint(0, len(annotations['images']))
        # take records by index
        image_record = annotations['images'][image_index]
        annotation_records = [annotation for annotation in annotations['annotations'] if
                              annotation['image_id'] == image_record['id']]
        image_path = f'{image_record["file_name"]}'
        image = Image.open(image_path)

        bboxes = [annotation_record['bbox'] for annotation_record in annotation_records]
        bboxes = np.array(bboxes)
        category_ids = [annotation_record['category_id'] for annotation_record in annotation_records]

        # draw:
        category_names = [category_names_table[category_id] for category_id in category_ids]
        draw_bbox_xywh(image, bboxes, category_names)
        return image

