import sys
sys.path.append(r'D:\project\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin\deep_sort_pytorch')
sys.path.append(r'D:\project\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin')

from yolov5_obb.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from detect_utils import rbox_to_poly, xyxy_to_xywh
from app.app import sim_app
import torch
import numpy as np


class ExtractorWorker:
    def __init__(self, opt):
        self.opt = opt
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt['config_deepsort'])
        self.deepsort = DeepSort(opt['device'], opt['deep_sort_weights'],
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_extractor=True)

    def get_features(self, frame_id, timestamp, det, img):
        t5 = time_synchronized()

        aabb_xywh_bboxes = []
        confs = []
        attrs = []
        bbox_xywh = None
        bbox_tlwh = None
        bbox_theta = None
        features = None
        confss = None

        if det is not None and len(det) > 0:
            for *rbox, conf, cls in reversed(det):  # 翻转list的排列结果,改为类别由小到大的排列
                rbox = np.array([r.numpy() for r in rbox])
                poly = np.int0(rbox_to_poly(rbox))
                obj = {"poly": poly, "conf": conf, "rbox": rbox}

                min_xy = np.min(poly, axis=0)
                max_xy = np.max(poly, axis=0)
                xyxy = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                aabb_xywh_bboxes.append(xywh_obj)
                attrs.append(obj)
            # Rescale boxes from padded_img_size to img size
            aabb_xywh = torch.Tensor(aabb_xywh_bboxes)

            # det:(num_nms_boxes, [xywhθ,conf,classid]) θ∈[0,179]
            # 翻转list的排列结果,改为类别由小到大的排列
            confss = det[:, 5]

            bbox_xywh, features, t56, t57, t58 = self.deepsort.get_features(aabb_xywh, img)

        t6 = time_synchronized()
        print('      [%4d] [Features]   %.3f s' % (frame_id, t6 - t5))

        sim_app()._timeline_.append(dict(frame=frame_id, begin=t5, end=t6, type="extract"))

        return bbox_xywh, features, confss, attrs
