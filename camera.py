import face_recognition
import cv2
from sklearn.externals import joblib
import lbp

# Load the classifier
clf = joblib.load("train_model.m")

# Get a reference to web cam
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
choice = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if choice:
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)

    choice = not choice

    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        name = 'No...'

        image = frame[top:bottom, left:right]
        resize_image = cv2.resize(image, (70, 70), interpolation=cv2.INTER_CUBIC)
        read_lbp = [lbp.get_vector(resize_image)]

        result = clf.predict(read_lbp)
        print(result)
        if result == [1]:
            name = 'Yes!!!'

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 10, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()