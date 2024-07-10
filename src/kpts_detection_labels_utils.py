import numpy as np
def create_detection_kpts_entries(images_bboxes, images_polygons, images_sizes, images_class_ids):

    """
    Description: one *.txt file per image,  one row per object, row format: class polygon vertices (x0, y0.....xn,yn)
    normalized coordinates [0 to 1].
    zero-indexed class numbers - start from 0


    :param images_paths: list of dataset image filenames
    :param images_polygons: list of per image polygons arrays
    :param images_class_ids:  list of per image class_ids
    :param output_dir: output dir of labels text files
    :return:
    """

    # detection_entries = create_detection_entries(images_bboxes, images_sizes, images_class_ids)
    entries=[]
    for image_polygons, images_size, class_ids, image_detection_entries in zip(images_polygons, images_sizes,
                                                              images_class_ids, images_bboxes):
        image_detection_entries=np.array(image_detection_entries)
        image_polygons=np.array(image_polygons)

        im_height = images_size[0]
        im_width = images_size[1]

        img_kpts = (image_polygons/np.array([im_width, im_height]))
        # concat valid field:
        img_kpts_valid = np.full( [img_kpts.shape[0], img_kpts.shape[1], 1], 2.) # shape: [nobj, nkpts, 1]
        img_kpts = np.concatenate([img_kpts, img_kpts_valid], axis=-1).reshape(img_kpts.shape[0], -1) # flatten kpts per object

        img_entries=[]
        for detection_entry, kpts   in zip(image_detection_entries, img_kpts):
            entry = f"{detection_entry} {' '.join(str(kpt) for kpt in list(kpts.reshape(-1)))}"
            img_entries.append(entry)
        entries.append(img_entries)
    return entries

