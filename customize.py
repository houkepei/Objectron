# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# img = '../demo/demo.jpg'
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result)

import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from numpy import loadtxt
import cv2

img_idx = '000072'
img_url = f'/home/ANT.AMAZON.COM/cgzhang/PycharmProjects/datasets/kitti_tiny/training/image_2/{img_idx}.jpeg'
label_url = f'/home/ANT.AMAZON.COM/cgzhang/PycharmProjects/datasets/kitti_tiny/training/label_2/{img_idx}.txt'
cal_url = f'/home/ANT.AMAZON.COM/cgzhang/PycharmProjects/datasets/kitti/training/calib/{img_idx}.txt'

# load image
image = mmcv.imread(img_url, flag='color', channel_order='rgb')
height, width = image.shape[:2]

# load annotations
lines = mmcv.list_from_file(label_url)
content = [line.strip().split(' ') for line in lines]
# bbox_names = [x[0] for x in content]
loc_3d = [[float(info) for info in x[11:14]] for x in content if x[0] != "DontCare"]
height_width_length = [[float(info) for info in x[8:11]] for x in content if x[0] != "DontCare"]
ry = [float(x[-1]) for x in content if x[0] != "DontCare"]

# load intrinsic matrix
lines = mmcv.list_from_file(cal_url)
content = [line.strip().split(' ') for line in lines]
cam = content[2]
assert cam[0][:2] == 'P2'
proj_mat = loadtxt(cam[1:]).reshape(3, 4)

# Radius of circle
radius = 4
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

num_objects = len(loc_3d)
fig, ax = plt.subplots(num_objects, 1, figsize=(12, 16))
edges = (
    [1, 2], [4, 3], [5, 6], [8, 7],  # lines along x-axis
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along y-axis
    [1, 4], [5, 8], [2, 3], [6, 7]  # lines along z-axis
)

for i, ([x, y, z], [h, w, l], ry) in enumerate(zip(loc_3d, height_width_length, ry)):
    img = image.copy()
    R = np.array([np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)]).reshape(3, -1)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    eight_corners = []
    colorsIndex = 1
    for cx, cy, cz in zip(x_corners, y_corners, z_corners):
        pts = np.array([cx, cy, cz])
        pts = np.matmul(R, pts) + np.array([x, y, z])
        pts = np.matmul(proj_mat, np.append(pts, 1.0))
        pts[:2] /= pts[2]
        # print(pts[:2])
        eight_corners.append(pts[:2].astype(int))
        cv2.circle(img, [int(pts[0]), int(pts[1])], radius, color, thickness)
        cv2.putText(img, str(colorsIndex), (int(pts[0]), int(pts[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 5)
        colorsIndex = colorsIndex + 1
    # TODO: draw the connections between pts
    for e in edges:
        start_kp = eight_corners[e[0] - 1]
        end_kp = eight_corners[e[1] - 1]
        cv2.line(img, start_kp, end_kp, (255, 255, 0), 2)
    ax[i].grid(False)
    ax[i].imshow(img)
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)

fig.tight_layout()
plt.show()
