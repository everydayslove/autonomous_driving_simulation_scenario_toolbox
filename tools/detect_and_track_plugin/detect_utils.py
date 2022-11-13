import plotly.express as px
import pandas as pd
import queue


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
resized_range = 0.5

step_index = 2

queue_limit = 200

img_queue = queue.Queue(2)
detection_queue = queue.Queue(queue_limit)
feature_queue = queue.Queue(queue_limit)
tracked_obj_queue = queue.Queue(queue_limit)
interpolated_obj_queue = queue.Queue(queue_limit)
convert_coordinate_obj_queue = queue.Queue(queue_limit)
attribute_for_display_queue = queue.Queue(queue_limit)
attribute_for_stream_queue = queue.Queue(queue_limit)
streaming_output_queue = queue.Queue(queue_limit)
timeline = []

def dist2(p1, p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])

def dot(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

# def set_fov():
#     import usb.core
#     import usb.util
#     import array
#     dev = usb.core.find(idVendor=0x046d, idProduct=0x085e)
#     if dev is None:
#         raise ValueError('Device not found')
#
#     reattach = False
#     if dev.is_kernel_driver_active(0):
#         reattach = True
#         dev.detach_kernel_driver(0)
#         usb.util.claim_interface(dev, 0)
#
#     try:
#         ret = dev.ctrl_transfer(0x21, 0x01, 0x0500, 0x0a00, array.array('B', [1]))
#         ret = dev.ctrl_transfer(0xa1, 0x81, 0x0500, 0x0a00, 1)
#         print("fov: ", ret)
#     finally:
#         usb.util.release_interface(dev, 0)
#         if reattach:
#             dev.attach_kernel_driver(0)
#         usb.util.dispose_resources(dev)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# def poly_to_rbox(poly):
#     p = np.array(poly)
#     base_x = (p[0] + p[2] + p[4] + p[6]) / 4
#     base_y = (p[1] + p[3] + p[5] + p[7]) / 4
#     length = 0.5 * math.sqrt((p[0] - p[6]) ** 2 + (p[1] - p[7]) ** 2) + 0.5 * math.sqrt((p[2] - p[4]) ** 2 + (p[3] - p[5]) ** 2)
#     width = 0.5 * math.sqrt((p[0] - p[2]) ** 2 + (p[1] - p[3]) ** 2) + 0.5 * math.sqrt((p[6] - p[4]) ** 2 + (p[5] - p[7]) ** 2)
#     if length < width:
#         length, width = width, length
#     theta = math.degrees(math.atan2(length, width))
#     return [base_x, base_y, length, width, theta]

def poly_to_rbox(poly):
    p = np.array(poly)
    base_x = (p[0] + p[2] + p[4] + p[6]) / 4
    base_y = (p[1] + p[3] + p[5] + p[7]) / 4
    length = 0.5 * math.sqrt((p[0] - p[6]) ** 2 + (p[1] - p[7]) ** 2) + 0.5 * math.sqrt((p[2] - p[4]) ** 2 + (p[3] - p[5]) ** 2)
    width = 0.5 * math.sqrt((p[0] - p[2]) ** 2 + (p[1] - p[3]) ** 2) + 0.5 * math.sqrt((p[6] - p[4]) ** 2 + (p[5] - p[7]) ** 2)
    if length < width:
        length, width = width, length
        theta = math.degrees(math.atan2(p[1] + p[3] - p[7] - p[5], p[0] + p[2] - p[6] - p[4])) + 90
    else:
        theta = math.degrees(math.atan2(p[1] + p[3] - p[7] - p[5], p[0] + p[2] - p[6] - p[4]))
    return [base_x, base_y, length, width, theta]


def rbox_to_poly(rbox):
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()
    base = np.array([rbox[0], rbox[1]])
    angle = -math.radians(rbox[4])
    dir = np.array([math.cos(angle), math.sin(angle)])
    left = np.array([-math.sin(angle), math.cos(angle)])
    length = rbox[2] * 0.5
    width = rbox[3] * 0.5
    poly = np.array([
        base + dir * length + left * width,
        base + dir * length - left * width,
        base - dir * length - left * width,
        base - dir * length + left * width])
    return poly

def rbox_to_heading_arrow(rbox):
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()
    base = np.array([rbox[0], rbox[1]])
    angle = -math.radians(rbox[4])  # FIXME
    dir = np.array([math.cos(angle), math.sin(angle)])
    left = np.array([-math.sin(angle), math.cos(angle)])
    length = rbox[2] * 0.5
    width = rbox[3] * 0.5
    poly = np.array([
        base + dir * length + left * width,
        base + dir * length - left * width,
        base + dir * length * 1.5])
    return np.int0(poly)

def plot_one_rotated_box(id, poly, img, color, label=None, line_thickness=2):
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=color, thickness=line_thickness)
    if label:
        pt = (int(poly[0][0]), int(poly[0][1]))
        color = compute_color_for_labels(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, pt, (pt[0] + t_size[0] + 3, pt[1] + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (pt[0], pt[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)


def my_process_dataframe_timeline(args):
    """
    Massage input for bar traces for px.timeline()
    """
    args["is_timeline"] = True
    if args["x_start"] is None or args["x_end"] is None:
        raise ValueError("Both x_start and x_end are required")

    x_start = args["data_frame"][args["x_start"]]
    x_end = args["data_frame"][args["x_end"]]

    # note that we are not adding any columns to the data frame here, so no risk of overwrite
    args["data_frame"][args["x_end"]] = (x_end - x_start)
    args["x"] = args["x_end"]
    del args["x_end"]
    args["base"] = args["x_start"]
    del args["x_start"]
    return args

def write_timeline(tt0):
    px._core.process_dataframe_timeline = my_process_dataframe_timeline

    summary = OrderedDict()
    for d in timeline:
        d["begin"] = d["begin"] - tt0
        d["end"] = d["end"] - tt0
        type = d["type"]
        if type in summary:
            summary[type] += d["end"] - d["begin"]
        else:
            summary[type] = d["end"] - d["begin"]

    print("Summary:")
    for stage, time in summary.items():
        print("\t%10s : %6.2fs" % (stage, time))

    df = pd.DataFrame(timeline)

    fig = px.timeline(df, x_start="begin", x_end="end", y="frame", color="type")
    fig.layout.xaxis.type = 'linear'
    #fig.data[0].x = df.delta.tolist()
    fig.update_yaxes(autorange="reversed")
    #fig.show()
    fig.write_html("timeline.html")


def simple_read_from_video(cap):
    ret_val, img = cap.read()
    if img is None:
        return False, None, None

    if not ret_val:
        print("Cap read error code = ", ret_val)
        return False, None, None
    padded_img = letterbox(img, new_shape=args.img_size)[0]
    # Convert
    padded_img = padded_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    padded_img = np.ascontiguousarray(padded_img)
    return True, img, padded_img