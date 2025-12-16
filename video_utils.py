import cv2 as cv

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def resize_live_video(video, width=480, height=480):
    video.set(3, width)
    video.set(4, height)
