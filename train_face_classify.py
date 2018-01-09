import cv2
import os
import numpy as np
from recognition.facenet import get_dataset
from recognition.FaceRecognition import FaceRecognition
from detection.FaceDetector import FaceDetector
from classifier.FaceClassifier import FaceClassifier

datadir = './media/train_classifier'

face_detector = FaceDetector()
face_recognition = FaceRecognition()
face_classfier = FaceClassifier()

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [dataset[i].name] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

dataset = get_dataset(datadir)
paths, labels = get_image_paths_and_labels(dataset)
print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))

# Run forward pass to calculate embeddings
print('Calculating features for images')
image_size = 160
nrof_images = len(paths)
features = np.zeros((2*nrof_images, 128))
labels = np.asarray(labels).repeat(2)
for i in range(nrof_images):
    img = cv2.imread(paths[i])
    if img is None:
        print('Open image file failed: ' + paths[i])
        continue
    boxes, scores = face_detector.detect(img)
    if len(boxes) < 0 or scores[0] < 0.5:
        print('No face found in ' + paths[i])
        continue

    cropped_face = img[boxes[0][0]:boxes[0][2], boxes[0][1]:boxes[0][3], :]
    cropped_face_flip = cv2.flip(cropped_face,1)
    features[2*i,:] = face_recognition.recognize(cropped_face)
    features[2*i+1,:] = face_recognition.recognize(cropped_face_flip)

print('Start training for images')
face_classfier.train(features, labels, model='svm', save_model_path='./classifier/trained_classifier.pkl')

