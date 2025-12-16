import cv2 as cv
from src.face_encoding import load_known_face
from src.video_utils import resize_live_video
from src.face_detection import load_face_detector, detect_faces
from src.face_recognition_logic import recognize_faces

# تحميل الوجه المعروف
known_encoding, known_name = load_known_face("data/known_face.jpg", "Yaser")

# تحميل نموذج كشف الوجه
face_detector = load_face_detector("models/haarcascade_frontalface_default.xml")

# فتح الكاميرا
video = cv.VideoCapture(0)
if not video.isOpened():
    print("Cannot open camera")
    exit()

resize_live_video(video)

frame_count = 0
last_faces = []
N = 60  # نفذ التعرف كل N فريم

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if frame_count % N == 0:
        faces = detect_faces(face_detector, gray)
        last_faces = recognize_faces(
            frame,
            faces,
            known_encoding,
            known_name
        )

    for (x, y, w, h, name) in last_faces:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label_y = max(y - 25, 0)
        cv.rectangle(frame, (x, label_y), (x+w, y), color, cv.FILLED)
        cv.putText(frame, name, (x+5, y-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
