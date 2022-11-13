class TrackingWorker:
    def __init__(self, opt):
        self.opt = opt
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        self.deepsort = DeepSort(opt.device, cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)
        self.t_a = None
        self.t_b = None


    def run(self):
        while True:
            frame_id, timestamp, bbox_xywh, features, confss, attrs, img = feature_queue.get()
            if img is None:
                break
            timestamp, output, attrs, img = self.track(frame_id, timestamp, bbox_xywh, features, confss, attrs, img)
            tracked_obj_queue.put((frame_id, timestamp, output, attrs, img))
        tracked_obj_queue.put((None, None, None, None, None))

    def track(self, frame_id, timestamp, bbox_xywh, features, confss, attrs, img):
        t7 = time_synchronized()

        self.deepsort.height, self.deepsort.width = img.shape[:2]

        if bbox_xywh is None or len(bbox_xywh) == 0:
            self.deepsort.increment_ages()
            outputs = []
        else:
            outputs, attrs, t77, t78 = self.deepsort.update(bbox_xywh, features, confss, attrs)

        t8 = time_synchronized()

        print('      [%4d] [DeepSORT]   %.3f s' % (frame_id, t8 - t7))

        global timeline
        # timeline.append(dict(frame=frame_id, begin=t7, end=t77, type="track_A"))
        # timeline.append(dict(frame=frame_id, begin=t77, end=t78, type="track_B"))
        # timeline.append(dict(frame=frame_id, begin=t78, end=t8, type="track_C"))
        timeline.append(dict(frame=frame_id, begin=t7, end=t8, type="track"))

        # self.t_a = self.t_b
        # self.t_b = time_synchronized()
        # if self.t_a is not None:
        #     print('[%3d] [%4d] [Track]      %.3f s  ====\n' % (len(outputs), frame_id, self.t_b - self.t_a))
        print('[%3d] [%4d] [Track]      \n' % (len(outputs), frame_id))
        return timestamp, outputs, attrs, img