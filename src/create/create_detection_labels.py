from src.create.create_bboxes import CreateBboxes

class CreateDetectionLabels(CreateBboxes):
    def __init__(self, iou_thresh, bbox_margin):
        super().__init__(iou_thresh, bbox_margin)

    def create_detection_labels(self, batch_bboxes, batch_categories_ids):
        batch_labels = []
        for image_boxes, image_categories_ids in zip(batch_bboxes, batch_categories_ids):
            entries = [f"{category_id} {' '.join(str(vertex) for vertex in list(bbox))} " for bbox, category_id in
                       zip(image_boxes, image_categories_ids)]
            batch_labels.append(entries)
        return batch_labels

    def run(self, batch_polygons, batch_image_size, batch_categories_ids):
        batch_bboxes = self.create_batch_bboxes(batch_polygons, batch_image_size)
        batch_labels = self.create_detection_labels(batch_bboxes, batch_categories_ids)
        return batch_labels
