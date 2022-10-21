import cv2
import ctypes
from ctypes import *
import numpy as np


# 加载共享链接库
# matso = ctypes.cdll.LoadLibrary("build/libMatSo.so")
matso = ctypes.windll.LoadLibrary(r"F:\work\workSpace\minanqiang\imageregistration_new\image_registration\build\Debug\ShakeImage.dll")

# matso中有两个函数我们会使用到
# 现在对这两个函数定义入参和出参的类型
# 参考：https://blog.csdn.net/qq_40047008/article/details/107785856
matso.get_mat_and_return_uchar.argtypes = (POINTER(c_ubyte), c_int, c_int, c_int)
matso.get_mat_and_return_uchar.restype = POINTER(c_ubyte)


matso.LoadFrame.argtypes = (POINTER(c_ubyte), POINTER(c_ubyte), c_int, c_int, c_int)
matso.LoadFrame.restype = POINTER(c_ubyte)
matso.ReleaseFrame.argtypes = (POINTER(c_ubyte),)
    # cv2.imshow("q", np_canny)
    # cv2.waitKey(0)

import cv2
from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np

# dll = ctypes.WinDLL('MyDLL.dll')

# img = cv2.imread('input.png')
# ptr, canny = cpp_canny(img)
# cv2.imshow('canny', canny)
# cv2.waitKey(2000)
# def cpp_canny(input):
#     if len(img.shape) >= 3 and img.shape[-1] > 1:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     h, w = gray.shape[0], gray.shape[1]
#
#     # 获取numpy对象的数据指针
#     frame_data = np.asarray(gray, dtype=np.uint8)
#     frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
#
#     # 设置输出数据类型为uint8的指针
#     dll.cpp_canny.restype = ctypes.POINTER(ctypes.c_uint8)
#
#     # 调用dll里的cpp_canny函数
#     pointer = dll.cpp_canny(h, w, frame_data)
#
#     # 从指针指向的地址中读取数据，并转为numpy array
#     np_canny = np.array(np.fromiter(pointer, dtype=np.uint8, count=h * w))
#
#     return pointer, np_canny.reshape((h, w))



# # 将内存释放
# dll.release(ptr)


def printbuf(data, len):
    p = (c_ubyte * len)()
    for i in range(len):
        p[i] = data[i]

    b = bytes(bytearray(p))
    print(b)


def read_video(path):
    video = cv2.VideoCapture(path)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame = 0
    ret = video.isOpened()
    first_gary_img = None
    last_homography = np.eye(3)
    # n = np.array(
    #     [[[1, 2, 3, 4], [5, 6, 7, 8]], [[10, 11, 12, 14], [15, 16, 17, 18]], [[11, 12, 43, 32], [1, 5, 10, 23]]],
    #     np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # vid_writer = cv2.VideoWriter(path + rf'\{dir}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (img.shape[1], img.shape[0]))
    frame_size = (int(video_width), int(video_height))
    # frame_size = (100, 100)
    output = cv2.VideoWriter(r"F:\img\output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    # out = cv2.VideoWriter(r"F:\img\output.avi", fourcc, fps, (video_width, video_height))
    while ret:
        ret, current_img = video.read()
        if ret is False:
            break
        frame += 1
        if frame == 1:
            first_gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("F:\\1.jpg", first_gray_img)
            continue
        # if frame <= 45:
        #     continue
        # frame_height = cv.get(cv2.CAP_PEOP_FRAME_HEIGHT)
        # frame_width = cv.get(cv2.CAP_PEOP_FRAME_WIDTH)

        # height = current_img.cols
        # width = current_img.rows
        # warpPerspective(image_2, forecast, lastHomography, size);

        # img_copy = current_img.co()
        height, width, channels = current_img.shape

        forecast_img = cv2.warpPerspective(current_img, last_homography, (width, height))
        current_gray_img = cv2.cvtColor(forecast_img, cv2.COLOR_BGR2GRAY)
        # rows, cols, channels = current_img.shape
        # img = cv2.imread("face.jpeg")
        # _, _, channels = current_img.shape
        c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        first_gray_img_data = first_gray_img.ctypes.data_as(c_ubyte_p)  # 将ndarray转为c++的uchar类型
        # first_gray_img_data = matso.get_mat_and_return_uchar(first_gray_img_data, height, width, 1)

        current_gray_img_data = current_gray_img.ctypes.data_as(c_ubyte_p)  # 将ndarray转为c++的uchar类型
        # current_gray_img_data = matso.get_mat_and_return_uchar(current_gray_img_data, height, width, 1)
        # current_gray_img_data = current_gray_img.data.ctypes.data_as(POINTER(c_ubyte))  # 将ndarray转为c++的uchar类型
        # first_gray_img_data = first_gray_img.data
        # current_gray_img_data = current_gray_img.data
        # printbuf(first_gray_img_data, 512)
        # printbuf(current_gray_img_data, 512)
        homography_data = matso.LoadFrame(first_gray_img_data, current_gray_img_data, height, width, 1)  # 调用链接库函数，得到uchar*数据
        # printbuf(homography_data, 512)
        # p = (c_ubyte * 36)()
        # for i in range(36):
        #     p[i] = homography_data[i]
        #
        # b = bytes(bytearray(p))
        # print(b)

        # 注意这里的rows、rows和channels是C++函数中返回时的Mat尺寸，与上面中不是同一个意思
        # 但是因为上面传入函数并返回过程中没有改变图像的shape，所以在数值上是一样的
        # np_canny = np.array(np.fromiter(return_uchar_data, dtype=np.uint8, count=cols * rows * 1))
        homography_float = cast(homography_data, POINTER(c_float))

        homography = np.fromiter(homography_float, dtype=np.float32, count=9)
        homography = homography.reshape((3, 3))
        # POINTER(c_ubyte)
        # np_canny = np.array(np.fromiter(return_uchar_data, dtype=np.float32, count=9))
        # np_canny = np_canny.reshape((3, 3))
        print("before last_homography:{}".format(last_homography))
        print("before homography:{}".format(homography))

        # import numpy as np

        # arr1 = np.array([[1, 2],
        #                  [3, 4]])
        # arr2 = np.array([[5, 6],
        #                  [7, 8]])

        arr_result = np.matmul(homography, last_homography)
        matso.ReleaseFrame(homography_data)
        # print(arr_result)

        # tmp = homography * last_homography
        tmp = np.matmul(homography, last_homography)
        last_homography = tmp
        print("after last_homography:{}".format(last_homography))
        # last_homography = homography*last_homography
        # print(last_homography)
        result_image = cv2.warpPerspective(forecast_img, last_homography, (width, height))
        file_name = "{}.jpg".format(frame)
        cv2.imwrite(r"F:\img\r_{}".format(file_name), result_image)
        # cv2.imwrite(r"F:\img\{}".format(file_name), result_image)
        # cv2.imwrite(r"F:\img\first_{}".format(file_name), first_gray_img)
        # cv2.imwrite(r"F:\img\cur_{}".format(file_name), current_gray_img)
        output.write(result_image)
        print("{}".format(file_name))
        # print(np_canny)


if __name__ == "__main__":
    read_video(r"F:\2022-09-24 16-24-54.mkv")