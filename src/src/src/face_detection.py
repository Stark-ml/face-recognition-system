import cv2 as cv

def load_face_detector(model_path):
    return cv.CascadeClassifier(model_path)

def detect_faces(detector, gray_frame):
    return detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )
