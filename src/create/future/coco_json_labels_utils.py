import json
def create_coco_json_lable_files(images_paths, images_sizes, images_bboxes, images_class_ids,
                   category_names,  category_ids, annotations_output_path):
    """
     :param images_paths: list of dataset image filenames
    :param images_sizes: list of per image [im.height, im.width] tuples
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_class_ids: list of per image class_ids arrays
    :param category_names: list of all dataset's category names
    :param category_ids:  list of all dataset's category ids

    :param annotations_output_path: path for output file storage
    :return:
    """

    print('create_coco_labels_file')

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    # map_categories_id = {}

    # fill category
    for category_name, category_id in zip(category_names, category_ids):

        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": category_names,
                "id": category_id,
                "name": category_name,
            })
            added_category_names.append(category_name)

    images_records = []
    annotatons_records = []
    for example_id, (image_path, image_size, bboxes, categories_list) in enumerate(
            zip(images_paths, images_sizes, images_bboxes, images_class_ids)):

        # images records:

        images_records.append({
            "license": '',
            "file_name": image_path,
            "coco_url": "",
            'width': image_size[1],
            'height': image_size[0],
            "date_captured": str(datetime.now()),
            "flickr_url": "",
            "id": example_id
        })

        # annotatons_records
        for bbox, category_id in zip(bboxes, categories_list):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": category_id,
                "id": anno_id
            }
            )
            anno_id += 1
    date_today = date.today()
    info = {
        "description": " Dataset",
        "url": '',
        # "version": config.get('version', 1.0),
        "year": date_today.year,
        "contributor": '',
        "date_created": str(date_today),
        "licenses": '',
        "categories": categories_records
    }
    output_records = {
        "info": info,
        "licenses": [],
        "images": images_records,
        "categories": categories_records,
        "annotations": annotatons_records
    }
    print(f'Save annotation  in {annotations_output_path}')
    with open(annotations_output_path, 'w') as fp:
        json.dump(output_records, fp)

