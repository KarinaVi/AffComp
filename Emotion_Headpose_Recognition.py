import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/sm/Desktop/nodcontrol.avi', fourcc, 20.0, (640, 480))

# Part II: Emotion Detection
face_classifier=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel')
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# capture source video
cap = cv2.VideoCapture(2) # 0, 1, 2

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# path to facecascade
face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# function to get coordinates
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except:
        return int(p1[0][0]), int(p1[0][1])


# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX

# define movement thresholds
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 175

gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # number of frames a gesture is shown

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    old_gray = gray.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        face_center = x + w / 2, y + h / 3
        p0 = np.array([[face_center]], np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        print(p0)
        print(p1)
        cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # get the xy coordinates for points p0 and p1
            a, b = get_coords(p0), get_coords(p1)
            x_movement += abs(a[0] - b[0])
            y_movement += abs(a[1] - b[1])

            text = 'x_movement: ' + str(x_movement)
            if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
            text = 'y_movement: ' + str(y_movement)
            if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (0, 0, 255), 2)

            if x_movement > gesture_threshold:
                gesture = 'No'
            if y_movement > gesture_threshold:
                gesture = 'Yes'
            if gesture and gesture_show > 0:
                cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 1.2, (0, 0, 255), 3)
                gesture_show -= 1
            if gesture_show == 0:
                gesture = False
                x_movement = 0
                y_movement = 0
                gesture_show = 60  # number of frames a gesture is shown

            # print distance(get_coords(p0), get_coords(p1))
            p0 = p1

        else:
            cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()