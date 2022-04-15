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


# I'm running this Jupyter notebike locally. Manually import the objectron module.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
import objectron.dataset.box as Box

public_url = "https://storage.googleapis.com/objectron"
types = ["bottle",	"cereal_box", "chair", "cup"] #  bike	book	bottle	camera	cereal_box	chair	cup	laptop	shoe
for type in types:
    blob_path = public_url + "/v1/index/" + type + "_annotations_test"
    video_ids = requests.get(blob_path).text
    video_ids = video_ids.split('\n')
# Download the first ten videos in bike test dataset

    for i in range(1):
        video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
        metadata_filename = public_url + "/videos/" + video_ids[i] + "/geometry.pbdata"
        annotation_filename = public_url + "/annotations/" + video_ids[i] + ".pbdata"
        # video.content contains the video file.
        video = requests.get(video_filename)
        metadata = requests.get(metadata_filename)

        # Please refer to Parse Annotation tutorial to see how to parse the annotation files.
        annotation = requests.get(annotation_filename)

        file = open(type +"video.MOV", "wb")
        file.write(video.content)
        file.close()

        file = open(type+"geometry.pbdata", "wb")
        file.write(metadata.content)
        file.close()

        file = open(type+"annotation.pbdata", "wb")
        file.write(annotation.content)
        file.close()
