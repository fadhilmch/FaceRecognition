import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition
from classifier.FaceClassifier import FaceClassifier

face_detector = FaceDetector()
face_recognition = FaceRecognition()
face_classfier = FaceClassifier('./classifier/trained_classifier.pkl')
video_capture = cv2.VideoCapture(0)

print('Start Recognition!')
prevTime = 0
while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)  # resize frame (optional)

    curTime = time.time()  # calc fps
    find_results = []

    frame = frame[:, :, 0:3]
    boxes, scores = face_detector.detect(frame)
    face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
    face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
    print('Detected_FaceNum: %d' % len(face_boxes))

    if len(face_boxes) > 0:
        for i in range(len(face_boxes)):
            box = face_boxes[i]
            cropped_face = frame[box[0]:box[2], box[1]:box[3], :]
            cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)
            feature = face_recognition.recognize(cropped_face)
            name = face_classfier.classify(feature)

            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            # plot result idx under box
            text_x = box[1]
            text_y = box[2] + 20
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), thickness=1, lineType=2)
    else:
        print('Unable to align')

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, str, (text_fps_x, text_fps_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

