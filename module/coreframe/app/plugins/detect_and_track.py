# -*- coding: utf-8 -*-
import argparse
import os
from os import path
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import queue
import threading
from collections import OrderedDict
import datetime
import math
import asyncio
import ctypes
from shapely.geometry import Polygon, Point
import random

from abc import ABC
from app.base.base_plugin import BasePlugin
import json
import sys
# sys.path.insert(0, './yolov5_obb')
# from app.env import PATH_APP_ROOT
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(PATH_APP_ROOT)), 'tools'))
# sys.path.append(r'F:\work\workSpace\minanqiang\autonomous_driving_simulation_scenario_toolbox')
# sys.path.append(r'D:\project\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track\yolov5_obb')
project_path_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
tools_path = os.path.join(project_path_name, "tools")
detect_path = os.path.join(tools_path, "detect_and_track_plugin")
yolov5_path = os.path.join(detect_path, "yolov5_obb")
deep_sort_pytorch_path = os.path.join(detect_path, "deep_sort_pytorch")
sys.path.append(project_path_name)
sys.path.append(tools_path)
sys.path.append(detect_path)
sys.path.append(yolov5_path)
sys.path.append(deep_sort_pytorch_path)

# sys.path.append(r'F:\work\workSpace\minanqiang\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin')
# sys.path.append(r'F:\work\workSpace\minanqiang\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin\yolov5_obb')
# sys.path.append(r'F:\work\workSpace\minanqiang\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin\deep_sort_pytorch')

from yolov5_obb.utils.google_utils import attempt_download
from yolov5_obb.models.experimental import attempt_load
from yolov5_obb.utils.datasets import LoadImages, LoadStreams
from yolov5_obb.utils.general import check_img_size, scale_labels, xyxy2xywh, rotate_non_max_suppression
from yolov5_obb.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5_obb.utils.datasets import letterbox
from detect_utils import *
from camera_worker import CameraWorker
from detection_worker import DetectionWorker
from extractor_worker import ExtractorWorker
from tracking_worker import TrackingWorker
from app.base.configs import sim_cfg
from app.app import sim_app
from app.env import PATH_PROJECT_ROOT
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"


def get_class_name():
    return DetectAndTrack


deep_sort_file_name = "deep_sort.yaml"
ckpt_file_name = "ckpt.t7"


class DetectAndTrack(BasePlugin):
    def __init__(self, index):
        super(DetectAndTrack, self).__init__(index)
        self.resize_detection_range = 0.5 # the resized range of output detection video
        self.view_img = True
        self.save_detection_video = True
        self.cur_frame = None
        self.current_img = None
        self.fps = None
        self.total_frame = None
        self.video_file_name = None
        self.height = None
        self.width = None
        self.channels = None

    def init(self):
        # default parameters, can save into sim.conf file
        cur_path = os.path.dirname(os.path.abspath(__file__))

        app = sim_app()
        cfg = sim_cfg()

        device = ''
        with torch.no_grad():
            device = select_device(device)

        self.video_file_name = eval(cfg.common.video_file_path)
        model_path = str(cfg.get_str('detect::detect_model_path')[0])

        detect_args = dict()
        detect_args['detect_model_path'] =  eval(model_path)
        detect_args['conf_threshold'] = cfg.get_float('detect::object_confidence_threshold')[0]
        detect_args['iou_threshold'] = cfg.get_float('detect::iou_threshold')[0]
        detect_args['classes'] = [cfg.get_int('detect::classes', 0)[0]]
        detect_args['agnostic_nms'] = cfg.get_bool('detect::agnostic_nms', False)[0]
        detect_args['augment'] = cfg.get_bool('detect::augment', False)[0]
        detect_args['device'] = device

        self.detector = DetectionWorker(detect_args)

        track_extractor_args = {}
        deep_sort = os.path.join(cur_path, deep_sort_file_name)
        track_extractor_args['config_deepsort'] = deep_sort

        ckpt = os.path.join(cur_path, ckpt_file_name)
        track_extractor_args['deep_sort_weights'] = ckpt

        track_extractor_args['device'] = device
        self.extractor = ExtractorWorker(track_extractor_args)
        self.tracker = TrackingWorker(track_extractor_args)

        traj_name = path.basename(self.video_file_name).split('.')[0] + '_trajs.csv'
        self.traj_path = path.join(path.dirname(self.video_file_name), traj_name)
        if path.exists(self.traj_path):
            os.remove(self.traj_path)
        self.trajs = open(self.traj_path, 'w')
        self.trajs.write("ID,Time,PositionX,PositionY,Length,Width,Yaw,Category,PixelX,PixelY,\n")

        self.width = app._cv_Reader_._width_
        self.height = app._cv_Reader_._height_
        self.fps = app._cv_Reader_._fps_
        if self.save_detection_video:
            # codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            detection_video_name = path.join(path.dirname(self.video_file_name), path.basename(self.video_file_name).replace('.', '_detection.'))
            new_size = (int(self.width * self.resize_detection_range), int(self.height * self.resize_detection_range))
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(detection_video_name, codec, self.fps, new_size)

        self.start_time = time_synchronized()

        return True
        # raise NotImplementedError

    def run(self, parent, data):
        self.cur_frame = data['header']['cur_frame']
        self.current_img = data['image']

        print("processing the {} image\n".format(self.cur_frame))
        timestamp = float(self.cur_frame) / self.fps

        padded_img = self.get_padded_img(self.current_img)

        det = self.detector.detect(self.cur_frame, timestamp, padded_img, self.current_img)
        bbox_xywh, features, confss, attrs = self.extractor.get_features(self.cur_frame, timestamp, det, self.current_img)
        outputs, attrs = self.tracker.track(self.cur_frame, timestamp, bbox_xywh, features, confss, attrs, self.current_img)

        if self.view_img:
            resized_img = show(timestamp, outputs, attrs, self.current_img)
            cv2.imshow("result", resized_img)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        if self.save_detection_video:
            resized_img = show(timestamp, outputs, attrs, self.current_img)
            self.video_writer.write(resized_img)
        #
        # if len(outputs) > 0:
        #     for rbox in outputs:
        #         data_field = {
        #             "ID": int(rbox[5]),
        #             "TimeStamp": timestamp,
        #             "PixelX": round((rbox[0]), 4),
        #             "PixelY": round(rbox[1], 4),
        #             "Length": round(rbox[2], 4),
        #             "Width": round(rbox[3], 4),
        #             "Yaw": round(rbox[4], 4),
        #             "Category": "vehicle",
        #         }
        #         self.trajs.write(','.join([str(e) for e in data_field.values()]) + '\n')

    def get_padded_img(self, org_img):
        padded_img = letterbox(org_img, self.detector.width)[0]
        # Convert
        padded_img = padded_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img

    def finial(self):
        self.trajs.close()
        if self.save_detection_video:
            self.video_writer.release()

        print('Done %.3fs' % (time_synchronized() - self.start_time))
        return True