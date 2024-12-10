import numpy as np
from ultralytics import YOLO


class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), conf=0.6, save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []

        for seg in result.masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(3)

        return bboxes, segmentation_contours_idx, scores
