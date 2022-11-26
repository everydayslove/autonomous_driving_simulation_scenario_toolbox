import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']

import time
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class DeepSort(object):
    def __init__(self, device, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_extractor=False):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        if use_extractor:
            self.extractor = Extractor(model_path, device)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def get_features(self, bbox_xywh, img):
        self.height, self.width = img.shape[:2]
        # generate detections
        #t66 = time_synchronized()
        features, t56, t57, t58 = self._get_features(bbox_xywh, img)
        #print(bbox_xywh.shape, features.shape)
        # features = torch.zeros((bbox_xywh.shape[0], 512)) + 0.1
        # t56 = time_synchronized()
        # t57 = t56
        # t58 = t56
        #t67 = time_synchronized()
        #t68 = time_synchronized()
        return bbox_xywh, features, t56, t57, t58

    def update(self, bbox_xywh, features, confidences, attrs):
        detections = [Detection(bbox_xywh[i], conf, features[i], attrs[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]
        
        # update tracker
        self.tracker.predict()
        #t77 = time_synchronized()
        t77, t78 = self.tracker.update(detections)
        #t78 = time_synchronized()

        # output bbox identities
        outputs = []
        attrs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            box = track.to_xywh_theta()
            attr = track.attrs[-1]
            length = attr["rbox"][2]
            width = attr["rbox"][3]
            theta = attr["rbox"][4]
            outputs.append(np.array([box[0], box[1], length, width, theta, track.track_id], dtype=np.int))
            attrs.append(attr)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs, attrs, t77, t78

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, img):
        im_crops = []
        #t56 = time_synchronized()
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)

            # if x1 + 64 >= img.shape[1]:
            #     x1 = x2 - 64
            # else:
            #     x2 = x1 + 64
            # if y1 + 128 >= img.shape[0]:
            #     y1 = y2 - 128
            # else:
            #     y2 = y1 + 128

            # prevent zero-sized arrays
            if y1 == y2:
                if y2 + 1 == img.shape[0]:
                    y1 = y2 - 1
                else:
                    y2 = y1 + 1
            if x1 == x2:
                if x2 + 1 == img.shape[1]:
                    x1 = x2 - 1
                else:
                    x2 = x1 + 1
            im = img[y1:y2, x1:x2]
            im_crops.append(im)
        #t57 = time_synchronized()
        if im_crops:
            features, t56, t57, t58 = self.extractor(im_crops)
        else:
            features = np.array([])
        #t58 = time_synchronized()
        return features, t56, t57, t58
