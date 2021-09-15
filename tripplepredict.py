import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pyaudio
import wave
import threading
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tempfile

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/sm/Desktop/nodcontrol.avi', fourcc, 20.0, (640, 480))

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
UPDATE_SECONDS = 0.1
#WAVE_OUTPUT_FILENAME = "output.wav"
WAVE_OUTPUT_FILE = tempfile.SpooledTemporaryFile()


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def extract_feature_n(X, mfcc, chroma, mel):
    sample_rate=RATE
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("speechdata/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return np.array(x), y
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#DataFlair - Split the dataset
x_train,y_train=load_data(test_size=0.05)

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)

# Part II: Emotion Detection
face_classifier=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel')
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# capture source video
cap = cv2.VideoCapture(0) # 0, 1, 2

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


# find the face in the image
face_found = False
frame_num = 0
x = None
while frame_num < 30 or x is None:
    # Take first frame and find corners in it
    frame_num += 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_found = True
    cv2.imshow('image', frame)
    out.write(frame)
    cv2.waitKey(1)
face_center = x + w / 2, y + h / 3
p0 = np.array([[face_center]], np.float32)

gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # number of frames a gesture is shown


sound_counter = RATE * RECORD_SECONDS // CHUNK
sound_frames = []

sounds_prediction = ''

while True:

    if sound_counter == 0:
        """
        print('Saving sound thing?')
        wf = wave.open('output.wav', 'wb')
        #wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(sound_frames))
        print(sound_frames)
        wf.close()
        sound_frames = []
        sound_counter = RATE * RECORD_SECONDS // CHUNK
        print('Done saving?')
        """
        #feature=extract_feature(WAVE_OUTPUT_FILE, mfcc=True, chroma=True, mel=True)
        feature=extract_feature_n(np.hstack(sound_frames), mfcc=True, chroma=True, mel=True)
        sounds_prediction = model.predict([feature])[0]
        sound_frames = sound_frames[-(RATE * RECORD_SECONDS // CHUNK):]
        sound_counter = RATE * UPDATE_SECONDS // CHUNK

    sound_frames.append(np.frombuffer(stream.read(CHUNK), dtype=np.float32))
    sound_counter -= 1
    

    ret, frame = cap.read()
    old_gray = gray.copy()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
    cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    labels = []
    #old_gray = gray.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
       
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        else:
            cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)



    # get the xy coordinates for points p0 and p1
    a, b = get_coords(p0), get_coords(p1)
    x_movement += abs(a[0] - b[0])
    y_movement += abs(a[1] - b[1])

    text = 'x_position: ' + str(x_movement)
    if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
    text = 'y_position: ' + str(y_movement)
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

    cv2.putText(frame, 'Sound Emotion: ' + sounds_prediction, (50, 420), font, .8, (0, 0, 255), 2)

    if len(faces) > 0:
        face_center = x + w / 2, y + h / 3
        p0 = np.array([[face_center]], np.float32)
    else:
        p0 = p1

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()