import sys
sys.path.append(r'D:\project\autonomous_driving_simulation_scenario_toolbox\tools\detect_and_track_plugin\yolov5_obb')

from yolov5_obb.utils.google_utils import attempt_download
from yolov5_obb.models.experimental import attempt_load
from yolov5_obb.utils.datasets import LoadImages, LoadStreams
from yolov5_obb.utils.general import check_img_size, scale_labels, xyxy2xywh, rotate_non_max_suppression
from yolov5_obb.utils.torch_utils import select_device, time_synchronized
from app.app import sim_app

class DetectionWorker:
    def __init__(self, args):
        app = sim_app()
        self.width = app._cv_Reader_._width_
        self.device = '' # 'cuda device, i.e. 0 or 0,1,2,3 or cpu'
        self.detect_model_path = args['detect_model_path']
        self.conf_threshold = args['conf_threshold']
        self.iou_threshold = args['iou_threshold']
        self.classes = args['classes']
        self.agnostic_nms = args['agnostic_nms']
        self.augment = args['augment']
        
        # Load model
        # self.detect_model_path
        self.model = attempt_load(self.detect_model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.width = check_img_size(self.width, s=self.stride)  # check width
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.half()  # to FP16

        with torch.no_grad():
            self.device = select_device(self.device)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.width, self.width).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.t_a = None
        self.t_b = None

    def run(self):
        #cudnn.benchmark = True  # set True to speed up constant image size inference

        while True:
            frame_id, timestamp, padded_img, img = img_queue.get()
            if padded_img is None:
                break

            frame_id, timestamp, det, img = self.detect(frame_id, timestamp, padded_img, img)
            detection_queue.put((frame_id, timestamp, det, img))
        detection_queue.put((None, None, None, None))

    def detect(self, frame_id, timestamp, padded_img, ori_img):
        #return frame_id, None, ori_img

        t0 = time_synchronized()
        padded_img = torch.from_numpy(padded_img).to(self.device)
        padded_img = padded_img.half()
        padded_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if padded_img.ndimension() == 3:
            padded_img = padded_img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        pred = self.model(padded_img, augment=self.augment)[0]
        #pred = torch.zeros((1, 514080, 194))
        #t2 = time_synchronized()
        # Apply NMS
        pred = rotate_non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms, without_iouthres=False)
        #t3 = time_synchronized()

        # Process detections
        det = pred[0] # (num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
        if det is None:
            return frame_id, timestamp, None, ori_img
        det = det.cpu()

        det[:, :5] = scale_labels(padded_img.shape[2:], det[:, :5], ori_img.shape).round()

        t4 = time_synchronized()
        print('      [%4d] [Inference]  %.3f s' % (frame_id, t4 - t0))

        global timeline
        timeline.append(dict(frame=frame_id, begin=t0, end=t4, type="detect"))

        # self.t_a = self.t_b
        # self.t_b = time_synchronized()
        # if self.t_a is not None:
        #     print('      [%4d] [Detect]     %.3f s' % (frame_id, self.t_b - self.t_a))
        # print("=======================================================")
        return frame_id, timestamp, det, ori_img