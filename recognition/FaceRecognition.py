import os
import cv2
import numpy as np
import tensorflow as tf
from recognition import facenet

BASE_DIR = os.path.dirname(__file__) + '/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/20170512-110547.pb'
input_image_size = 160


class FaceRecognition:
    def __init__(self):
        # Load models
        self.recognition_graph = tf.Graph()
        self.sess = tf.Session(graph=self.recognition_graph)
        print('Loading feature extraction model')
        with self.sess.as_default():
            with self.recognition_graph.as_default():
                facenet.load_model(BASE_DIR + PATH_TO_CKPT)

    def __del__(self):
        self.sess.close()

    def recognize(self, image):

        images_placeholder = self.recognition_graph.get_tensor_by_name("input:0")
        embeddings = self.recognition_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.recognition_graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        emb_array = np.zeros((1, embedding_size))
        image = facenet.prewhiten(image)
        image = cv2.resize(image, (input_image_size, input_image_size), interpolation=cv2.INTER_AREA)
        image = image.reshape(-1, input_image_size, input_image_size, 3)
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array.squeeze()
