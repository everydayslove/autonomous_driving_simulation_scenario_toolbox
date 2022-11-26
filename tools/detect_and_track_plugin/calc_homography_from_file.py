# !/usr/bin/env python
import os.path

import cv2
import numpy as np
import os


def affine_point(affine_mat, point):
    p = np.append(point, 1)
    tmp = affine_mat.dot(p.T)
    return tmp[0]/tmp[2], tmp[1]/tmp[2]


def read_array_from_csv(file_name):
    f = open(file_name)
    output = []
    for line in f:
        x, y = line.split(',')
        p = [float(x), float(y)]
        output.append(p)
    return output


def calc_h():
    world_file = os.path.join(os.path.dirname(__file__), 'files/longyaoRoad_world')
    image_file = os.path.join(os.path.dirname(__file__), 'files/longyaoRoad_4_stable')
    print(world_file)
    print(image_file)
    worlds = np.array(read_array_from_csv(world_file))
    images = np.array(read_array_from_csv(image_file))

    # Calculate Homography
    h, status = cv2.findHomography(images, worlds)
    # print(status)
    print(h)
    # pts_image_to_verify = np.array([2603, 1304])      # [1371, 734]
    # print(affine_point(h, pts_image_to_verify))

    return h


if __name__ == '__main__':
    calc_h()



