import json
import numpy as np
from numpy import loadtxt
import cv2
import matplotlib.pyplot as plt
import mmcv
import imageio

def parse_meta_data(filename, image_path,depth_path):
    image = mmcv.imread(image_path, flag='color', channel_order='rgb')
    height, width = image.shape[:2]
    print(height, width)

    with open(filename, "r") as read_file:
        data = json.load(read_file)

        cam = data["cam"]
        objs = data["objs"]
        FOV = cam["FOV"]
        FOV_VERTICAL = cam["FOV_VERTICAL"]
        focalLength = cam["focalLength"]

    # 0: x-axis, 1:y-axis
    sensor_shape = [np.tan(FOV / 2.0) * 2, 0]
    sensor_shape[1] = sensor_shape[0] * height / width
    sensor_shape = np.array(sensor_shape)
    mm2pixel = width / sensor_shape[0]

    mg = cam["mg"]
    R = np.array([mg["x"], mg["y"], mg["z"]])
    off = np.array(mg["off"])

    num_objects = len(objs)
    fig, ax = plt.subplots(num_objects, 1, figsize=(12, 16))
    #fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    depth_img = imageio.imread(depth_path)
    depth_img = np.mean(depth_img, axis=2)
    depth_range = [0, 500]
    depth_image = (1.0 - depth_img) * (depth_range[1]-depth_range[0]) + depth_range[0]

    print(depth_image.min(), depth_image.max())



    edges = (
        [0, 1], [2, 3], [4, 5], [6, 7],  # lines along x-axis
        [0, 4], [1, 5], [2, 6], [3, 7],   # lines along y-axis
        [0, 2], [1, 3], [4, 6], [5, 7]  # lines along z-axis
    )
    # depth_image = imageio.imread('/home/ANT.AMAZON.COM/cgzhang/PycharmProjects/render/test_base.exr', format="EXR-FI")
    for i, obj in enumerate(objs):
        img = image.copy()
        pointList = obj['pointList']
        colorsIndex = 1
        point_2d = []

        for point in pointList:
            # from world to camera
            pts = np.matmul(R, point - off)

            pts_2d = pts[:2] / pts[2]
            pts_2d[1] *= -1.0  # reverse y-axis
            pts_2d[:2] = (pts_2d[:2] + sensor_shape / 2.0) * mm2pixel  # from metric space to pixel space
            pts_2d = (pts_2d + 0.5).astype(int)
            point_2d.append(pts_2d)

            # TODO: we need depth to show occlusion
            if (pts_2d[0] >= width or pts_2d[0] < 0) or (pts_2d[1] >= height or pts_2d[1] < 0):
                invisible = True
            else:
                d = depth_image[int(pts_2d[1]), int(pts_2d[0])]
                invisible = False if np.linalg.norm(pts) < (d+1) else True
                print(colorsIndex, d, np.linalg.norm(pts))

            if invisible:
                cv2.circle(img, [int(pts_2d[0]), int(pts_2d[1])], 10, (255, 0, 0), 4)
            else:
                cv2.circle(img, [int(pts_2d[0]), int(pts_2d[1])], 10, (0, 255, 0), 4)

            # if colorsIndex == 6 or colorsIndex == 4 :
            #     # cv2.circle(img, [int(pts_2d[0]), int(pts_2d[1])], 10, (255, 0, 0), 4)
            #     # cv2.putText(img, str(colorsIndex), [int(pts_2d[0]), int(pts_2d[1])], cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 4)
            #     # print(colorsIndex, d, np.linalg.norm(pts))
            #     pass
            # else:
            cv2.putText(img, str(colorsIndex), [int(pts_2d[0]), int(pts_2d[1])], cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 4)

            colorsIndex = colorsIndex + 1
        for e in edges:

            start_kp = point_2d[e[0]]
            end_kp = point_2d[e[1]]
            cv2.line(img, start_kp, end_kp, (0, 0, 0), 2)

        ax[i].grid(False)
        ax[i].imshow(img)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()


# resolveJson("/Users/kepeihou/Desktop/render/meta.json")
parse_meta_data("/Users/kepeihou/Objectron/rendertest4/meta.json",
                "/Users/kepeihou/Objectron/rendertest4/test2.png",
                "/Users/kepeihou/Objectron/rendertest4/test2_zdepth.exr")