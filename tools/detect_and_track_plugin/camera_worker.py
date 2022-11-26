class CameraWorker:
    def __init__(self, opt):
        #set_fov()
        self.opt = opt
        self.source, self.imgsz = opt.source, opt.img_size
        self.webcam = self.source == '0' or self.source.startswith('rtsp') or self.source.startswith('http') or self.source.endswith('.txt')
        self.frame_idx = -1
        if self.webcam:
            self.dataset = LoadStreams(self.source, img_size=self.imgsz)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz)
        self.dataset_iter = iter(self.dataset)

    def grab(self):
        self.frame_idx += 1
        t0 = time_synchronized()
        path, padded_img, img, vid_cap = next(self.dataset_iter, (None, None, None, None))
        t1 = time_synchronized()
        print('      [%4d] [Camera]     %.3f s  ####' % (self.frame_idx, t1 - t0))

        global timeline
        timeline.append(dict(frame=self.frame_idx, begin=t0, end=t1, type="camera"))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        return fps, self.frame_idx, padded_img, img