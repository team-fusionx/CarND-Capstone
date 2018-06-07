from styx_msgs.msg import TrafficLight
import keras
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from keras.applications import resnet50
from keras.models import model_from_json
from keras import backend as K
import rospy
import numpy as np
import cv2
import tensorflow as tf
from light_classification.models import model_file_join
import os.path


class TLClassifier(object):
    def __init__(self):
        rospy.logwarn("TLClassifier __init__ begins")
        self.current_light = TrafficLight.UNKNOWN

        # Load the Model

        # Clear any prev Keras sessions to improve model load time
        rospy.logwarn("clear_session")
        tf.reset_default_graph()
        K.reset_uids()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        K.set_learning_phase(0)
        # K.clear_session()
        rospy.logwarn("loading model")

        # Load Keras model from arch/weights files to improve model load time
        # vs using keras.models load_model() function
        model_path = "light_classification/models/"
        model_name = "ResNet50-UdacityRealandSimMix-Best-val_acc-1.0"
        model_file = model_path + model_name
        model_arch_file = model_file + '.json'
        model_weights_file = model_file + '.h5'

        json_file = open(model_arch_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        rospy.logwarn("model architecture loaded from json")

        if os.path.isfile(model_weights_file):
            model.load_weights(model_weights_file)
        else:
            # Join the split model weights files before loading
            model_file_join.join_file(model_weights_file, 3)
            model.load_weights(model_weights_file)
        rospy.logwarn("model weights loaded from h5")

        # # Save the model as arch/weights files
        # # get the architecture as a json string
        # arch = model.to_json()
        # # save the architecture string to a file somehow, the below will work
        # with open(model_arch_file, 'w') as arch_file:
        #     arch_file.write(arch)
        #     # now save the weights as an HDF5 file
        # arch_file.close()
        # model.save_weights(model_weights_file)

        self.model = model
        self.model._make_predict_function()
        rospy.logwarn("model made_predict_function completed")
        self.graph = tf.get_default_graph()
        rospy.logwarn("model tf.get_default_graph completed")

        # # Initial (long time) prediction on zero blank image
        # np_final = np.zeros((1, 224, 224, 3))
        # yhat = model.predict(np_final)
        # rospy.logwarn("model.predict on blank completed")

        self.labels = np.array(['green', 'none', 'red', 'yellow'])
        self.resize_width = 224
        self.resize_height = 224
        rospy.logwarn('TL Classifier init complete.')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        np_image_data = np.asarray(image)
        np_final = np.expand_dims(np_image_data, axis=0)
#        np_final = np_final/255
        np_final = resnet50.preprocess_input(np_final.astype('float64'))
        t0 = rospy.Time.now()
        model = self.model
        with self.graph.as_default():
            yhat = model.predict(np_final)
        dt = rospy.Time.now() - t0
#        yhat = yhat / yhat.sum()
        yhat = yhat[0]
        y_class = yhat.argmax(axis=-1)
        labels = self.labels

        rospy.loginfo('%s (%.2f%%) : GPU time (s) : %f', labels[y_class],
                      yhat[y_class]*100, dt.to_sec())

        self.current_light = TrafficLight.UNKNOWN
        if (yhat[y_class] > 0.5):
            if y_class == 0:
                self.current_light = TrafficLight.GREEN
            elif y_class == 2:
                self.current_light = TrafficLight.RED
            elif y_class == 3:
                self.current_light = TrafficLight.YELLOW

        return self.current_light
