from src.create.create_polygons import CreatePolygons
from src.create.create_bboxes import CreateBboxes

class CreateDetectionEntries(CreatePolygons, CreateBboxes):
    def __init__(self, config, iou_thresh, bbox_margin):
        CreatePolygons.__init__(self, config)
        CreateBboxes.__init__(self, iou_thresh, bbox_margin)

    @staticmethod
    def create_detection_labels(batch_bboxes, batch_categories_ids):
        batch_labels = []
        for image_boxes, image_categories_ids in zip(batch_bboxes, batch_categories_ids):
            entries = [f"{category_id} {' '.join(str(vertex) for vertex in list(bbox))} " for bbox, category_id in
                       zip(image_boxes, image_categories_ids)]
            batch_labels.append(entries)
        return batch_labels

    def run(self, nentries):
        batch_image_size, batch_categories_ids, batch_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas = self.create_batch_polygons(
            nentries)
        batch_bboxes = self.create_batch_bboxes(batch_polygons, batch_image_size)
        batch_labels = self.create_detection_labels(batch_bboxes, batch_categories_ids)
        return batch_polygons, batch_labels, batch_objects_colors, batch_image_size
