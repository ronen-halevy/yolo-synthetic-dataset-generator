from PIL import Image
import numpy as np
import cv2



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



def draw_kpts_dataset_example(image_path, label_path):
    [im, bboxes, kpts] = read_kpts_dataset_entry(image_path, label_path)
    image=np.array(im)
    bboxes = bboxes * np.tile([im.width, im.height],[2])
    bboxes = np.concatenate([bboxes[:, 0:2] - bboxes[:, 2:4] / 2, bboxes[:, 0:2] + bboxes[:, 2:4] / 2], axis=-1)

    for bbox in bboxes:
        plot_one_box(bbox, image, line_thickness=3)

    steps=3 # each entry is [x,y,occlusiuon]
    skeleton=[]
    plot_skeleton_kpts(image, kpts, steps, skeleton)
    image = Image.fromarray(image)
    return image


