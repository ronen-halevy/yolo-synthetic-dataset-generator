import numpy as np
def create_detection_kpts_entries(batch_bboxes, batch_polygons, batch_sizes, batch_class_ids):

    """
    Description: one *.txt file per image,  one row per object, row format: class polygon vertices (x0, y0.....xn,yn)
    normalized coordinates [0 to 1].
    zero-indexed class numbers - start from 0

    :param images_paths: list of dataset image filenames
    :param batch_polygons: list of per image polygons arrays
    :param batch_class_ids:  list of per image class_ids
    :param output_dir: output dir of labels text files
    :return:
    """
    # detection_entries = create_detection_entries(batch_bboxes, batch_sizes, batch_class_ids)
    entries=[]
    for image_polygons, image_size, class_ids, image_bboxes in zip(batch_polygons, batch_sizes,
                                                              batch_class_ids, batch_bboxes):
        image_bboxes=np.array(image_bboxes)
        image_polygons=np.array(image_polygons)

        im_height = image_size[0]
        im_width = image_size[1]

        img_kpts = (image_polygons/np.array([im_width, im_height]))
        # concat valid field:
        img_kpts_valid = np.full( [img_kpts.shape[0], img_kpts.shape[1], 1], 2.) # shape: [nobj, nkpts, 1]
        img_kpts = np.concatenate([img_kpts, img_kpts_valid], axis=-1).reshape(img_kpts.shape[0], -1) # flatten kpts per object

        img_entries=[]

        for bbox, kpts   in zip(image_bboxes, img_kpts):
            bbox = ' '.join(str( round(vertex, 2)) for vertex in list(bbox))
            entry = f"{bbox} {' '.join(str( round(kpt, 2)) for kpt in list(kpts.reshape(-1)))}"
            img_entries.append(entry)
        entries.append(img_entries)
    return entries

