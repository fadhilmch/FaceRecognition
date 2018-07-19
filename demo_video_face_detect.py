import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector

VIDEO_INPUT_FILE = './media/test_video/Zidane_1.avi'
VIDEO_OUTPUT_FILE = './media/test_video_output/Zidane_Detection_1.avi'

face_detector = FaceDetector()
video_capture = cv2.VideoCapture(VIDEO_INPUT_FILE)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUTPUT_FILE, fourcc, 24.0, (int(video_capture.get(3)),int(video_capture.get(4))))

print('Start Recognition!')
prevTime = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()

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
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            # plot result idx under box
            text_x = box[1]
            text_y = box[2] + 20
            cv2.putText(frame, 'Score: %2.3f' % face_scores[i], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
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
    
    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()

