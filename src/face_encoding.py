import face_recognition

def load_known_face(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        raise ValueError("No face found in the reference image.")

    return encodings[0], name
