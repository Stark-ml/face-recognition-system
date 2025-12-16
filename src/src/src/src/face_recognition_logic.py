import cv2 as cv
import face_recognition

def recognize_faces(frame, faces, known_encoding, known_name, tolerance=0.5):
    results = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        rgb_face = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_face)
        name = "Unknown"

        if len(encodings) > 0:
            match = face_recognition.compare_faces(
                [known_encoding],
                encodings[0],
                tolerance=tolerance
            )
            if match[0]:
                name = known_name

        results.append((x, y, w, h, name))

    return results
