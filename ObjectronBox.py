import numpy as np
import os
import requests
import struct
import sys
import subprocess
import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
from IPython.core.display import display,HTML
import matplotlib.pyplot as plt

# I'm running this Jupyter notebook locally. Manually import the objectron module.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
import objectron.dataset.box as Box


def get_geometry_data(geometry_filename):
    sequence_geometry = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        frame_number = 0

        while i < len(proto_buf):
            # Read the first four Bytes in little endian '<' integers 'I' format
            # indicating the length of the current message.
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame_data = ar_metadata_protocol.ARFrame()
            frame_data.ParseFromString(message_buf)

            transform = np.reshape(frame_data.camera.transform, (4, 4))
            projection = np.reshape(frame_data.camera.projection_matrix, (4, 4))
            view = np.reshape(frame_data.camera.view_matrix, (4, 4))
            position = transform[:3, -1]

            current_points = [np.array([v.x, v.y, v.z])
                              for v in frame_data.raw_feature_points.point]
            current_points = np.array(current_points)

            sequence_geometry.append((transform, projection, view, current_points))
    return sequence_geometry


# %%
# sequence_geometry = get_geometry_data("/Users/kepeihou/geometry.pbdata")


# print(sequence_geometry)

# %%
def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []

        # Object instances in the world coordinate system, These are stored per sequence,
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))

        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)

            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append((keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append((keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d, keypoint_size_list, view, projection))

    return result, instances


# %%
# result, instances = get_frame_annotation("/Users/kepeihou/annotation.pbdata")





# %%
# frame_ids = [100, 105, 110, 115]
# frames = grab_frame("/Users/kepeihou/video.MOV",frame_ids)
# print(frames)
# %%
def project_points(points, projection_matrix, view_matrix, width, height):
    p_3d = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T
    p_3d_cam = np.matmul(view_matrix, p_3d)
    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)
    # Project the points
    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T

    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5) * width
    pixels[:, 1] = ((1 + y) * 0.5) * height
    pixels = pixels.astype(int)
    return pixels


# %%
# Grab some frames from the video file.


# types = ["bottle",	"cereal_box", "chair", "cup"] #  bike	book	bottle	camera	cereal_box	chair	cup	laptop	shoe
# num_frames = len(frame_ids)
# for type in types:

sequence_geometry = get_geometry_data("/Users/kepeihou/Objectron/cupgeometry.pbdata")
annotation_data, instances = get_frame_annotation("/Users/kepeihou/Objectron/cupannotation.pbdata")
# 打开视频
videoCapture = cv2.VideoCapture("/Users/kepeihou/Objectron/cupvideo.MOV")
# 获取帧率和大小
fps = videoCapture.get(cv2.CAP_PROP_FPS)

size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))





# 设置输出的视频信息（视频文件名，编解码器，帧率，大小）
videoWriter = cv2.VideoWriter("cupTestVideo.MOV", cv2.VideoWriter_fourcc("X","V","I","D"), fps, size)

# 读取视频文件，如果要读取的视频还没有结束，那么success接收到的就是True，每一帧的图片信息保存在frame中，通过write方法写到指定文件中
success, frame = videoCapture.read()
i = 0
while success:
    # 视频2d 点


    points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[i]
    num_instances = len(num_keypoints)
    points_2d = []
    for instance_id in range(num_instances):
        instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[instance_id]

        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation
        vertices_3d = instance_vertices_3d * instance_scale.T;
        # Homogenize the points
        vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
        # Transform the homogenious 3D vertices with the box transformation
        box_vertices_3d_world = np.matmul(box_transformation, vertices_3d_homg)

        # If we transform these vertices to the camera frame, we get the 3D keypoints in the annotation data
        # i.e. vertices_3d_cam == points_3d
        vertices_3d_cam = np.matmul(frame_view_matrix, box_vertices_3d_world)
        vertices_2d_proj = np.matmul(frame_projection_matrix, vertices_3d_cam)

        # Project the points
        points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
        points2d_ndc = points2d_ndc.T

        # Convert the 2D Projected points from the normalized device coordinates to pixel values
        x = points2d_ndc[:, 1]
        y = points2d_ndc[:, 0]
        points2d = np.copy(points2d_ndc)
        points2d[:, 0] = ((1 + x) * 0.5) * width
        points2d[:, 1] = ((1 + y) * 0.5) * height

        points_2d.append(points2d.astype(int))
        # points2d are the projected 3D points on the image plane.
        colors = [
                (166, 206, 227),
                (31, 120, 180),
                (178, 223, 138),
                (51, 160, 44),
                (251, 154, 153),
                (227, 26, 28),
                (253, 191, 111),
                (255, 127, 0),
                (202, 178, 214)

            ]
            # Visualize the boxes

        for instance_id in range(num_instances):
            colorsIndex = 0
            if (instance_id == 0):
                for kp_id in range(num_keypoints[instance_id]):
                    kp_pixel = points_2d[instance_id][kp_id, :]
                    cv2.circle(frame, (kp_pixel[0], kp_pixel[1]), 30, colors[colorsIndex], -1)
                    cv2.putText(frame, str(colorsIndex), (kp_pixel[0], kp_pixel[1]), cv2.FONT_HERSHEY_PLAIN, 5,
                                    (0, 0, 0), 12)
                    colorsIndex = colorsIndex + 1
                for edge in Box.EDGES:  # 这个  Box.EDGES 不懂  所有数据在 point_2d 里面
                    start_kp = points_2d[instance_id][edge[0], :]
                    end_kp = points_2d[instance_id][edge[1], :]
                    cv2.line(frame, (start_kp[0], start_kp[1]), (end_kp[0], end_kp[1]), (255, 0, 0), 2)
            else:
                for kp_id in range(num_keypoints[instance_id]):
                    kp_pixel = points_2d[instance_id][kp_id, :]
                    cv2.circle(frame, (kp_pixel[0], kp_pixel[1]), 30, (255, 0, 0), -1)
                    cv2.putText(frame, str(colorsIndex), (kp_pixel[0], kp_pixel[1]), cv2.FONT_HERSHEY_PLAIN, 5,
                                    (0, 0, 0), 12)
                    colorsIndex = colorsIndex + 1
                for edge in Box.EDGES:  # 这个  Box.EDGES 不懂  所有数据在 point_2d 里面
                    start_kp = points_2d[instance_id][edge[0], :]
                    end_kp = points_2d[instance_id][edge[1], :]
                    cv2.line(frame, (start_kp[0], start_kp[1]), (end_kp[0], end_kp[1]), (255, 0, 0), 2)

        videoWriter.write(frame)
        i = i + 1
        success, frame = videoCapture.read()



