import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

import time
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class Extractor(object):
    def __init__(self, model_path, device):
        self.net = Net(reid=True)
        self.device = device
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.half()#.fuse().eval()
        self.size = (128, 64)
        self.norm0 = transforms.ToTensor()
        self.norm1 = torch.nn.Sequential(
            #transforms.ToTensor(),
            transforms.Resize(self.size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        self.norm1 = torch.jit.script(self.norm1)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        #print([im.shape for im in im_crops])
        #print(type(im_crops[0]), im_crops[0].shape)
        t56 = time_synchronized()
        #im_batch = torch.cat([self.norm1(self.norm0(im)).unsqueeze(0) for im in im_crops], dim=0)
        im_batch = torch.cat([self.norm(im).unsqueeze(0) for im in im_crops], dim=0)
        #im_batch = torch.zeros((len(im_crops), 3, 128, 64))
        t57 = time_synchronized()
        #print(im_batch.shape)
        im_batch = im_batch.to(self.device)
        im_batch = im_batch.half()
        t58 = time_synchronized()
        return im_batch, t56, t57, t58

    def __call__(self, im_crops):
        #t56 = time_synchronized()
        im_batch, t56, t57, t58 = self._preprocess(im_crops)
        #t57 = time_synchronized()
        with torch.no_grad():
            features = self.net(im_batch)
            #features = torch.zeros((im_batch.shape[0], 512))
        #t58 = time_synchronized()
        return features.cpu().numpy(), t56, t57, t58


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
