import cv2



videoCapture = cv2.VideoCapture("/Users/kepeihou/Objectron/bookvideo.MOV")

#获取帧率和大小
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#设置输出的视频信息（视频文件名，编解码器，帧率，大小）
videoWriter = cv2.VideoWriter(
    "myTestVideo.avi",cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)

#读取视频文件，如果要读取的视频还没有结束，那么success接收到的就是True，每一帧的图片信息保存在frame中，通过write方法写到指定文件中
success,frame = videoCapture.read()
i = 1

while success:
    cv2.circle(frame, (100 + i,200 + i), 30, (255, 0, 0), -1)
    i = i+1
    videoWriter.write(frame)
    success, frame = videoCapture.read()
