# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, xywh, confidence, feature, attrs):
        self.xywh_theta = np.hstack((xywh.cpu().numpy(), attrs["rbox"][4]))      
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.attrs = attrs

    def to_tlwh(self):
        """Convert bounding box to format `(min x, min y, width, height)`
        """
        ret = self.xywh_theta.copy()
        ret[0] = ret[0] - ret[2] / 2.
        ret[1] = ret[1] - ret[3] / 2.
        return ret[:4]

    def to_xyah_theta(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height, theta)`, where the aspect ratio is `width / height`.
        """
        ret = self.xywh_theta.copy()
        ret[2] /= ret[3]
        return ret
    
