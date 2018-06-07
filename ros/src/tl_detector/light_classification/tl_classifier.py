from styx_msgs.msg import TrafficLight
# import keras
# from keras.utils.generic_utils import CustomObjectScope
# from keras.models import load_model
from keras.applications import resnet50
import rospy
import numpy as np
import cv2
import tensorflow as tf
from light_classification.models import model_file_join
import os

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class TLClassifier(object):
    def __init__(self):
        rospy.logwarn("TLClassifier __init__ begins")
        t_0 = rospy.Time.now()
        # Clear any prev Keras sessions to improve model load time
        rospy.logwarn("clear_session")


        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        model_name = "ResNet50-UdacityRealandSimMix-Best-val_acc-1.0"
        cwd = os.getcwd()
        graph_path = cwd + '/' + "light_classification/models/graph/"
        graph_file = graph_path + model_name + '.pb'

        if os.path.isfile(graph_file) is False:
            rospy.logwarn("combining graph file")
            model_file_join.join_file(graph_file, 3)

        t0 = rospy.Time.now()
        rospy.logwarn("loading graph {}".format(graph_file))
        self.graph = load_graph(graph_file)
        t1 = rospy.Time.now()
        dt1 = t1 - t0
        rospy.logwarn("graph loaded in %f s",dt1.to_sec())
        self.x = self.graph.get_tensor_by_name('prefix/image_input_1:0')
        self.y = self.graph.get_tensor_by_name('prefix/predictions_1/Softmax:0')
        rospy.logwarn("start TF session")
        t2 = rospy.Time.now()
        self.sess = tf.Session(graph=self.graph, config=config)
        t3 = rospy.Time.now()
        dt3 = t3 - t2
        rospy.logwarn("TF session started in %f s",dt3.to_sec())

        # Initial (long time) prediction on zero blank image
        np_final = np.zeros((1, 224, 224, 3))
        yhat = self.sess.run(self.y, feed_dict={self.x: np_final})
        rospy.logwarn("predict on blank completed")

        self.current_light = TrafficLight.UNKNOWN
        self.labels = np.array(['green', 'none', 'red', 'yellow'])
        self.resize_width = 224
        self.resize_height = 224
        t_1 = rospy.Time.now()
        dt_1 = t_1 - t_0
        rospy.logwarn('TL Classifier init completed in %f s.',dt_1.to_sec())

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        t0 = rospy.Time.now()
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        np_image_data = np.asarray(image)
        np_final = np.expand_dims(np_image_data, axis=0)
        np_final = resnet50.preprocess_input(np_final.astype('float64'))

        t1 = rospy.Time.now()
        dt1 = t1 - t0

        yhat = self.sess.run(self.y, feed_dict={self.x: np_final})

        dt2 = rospy.Time.now() - t1

        yhat = yhat[0]
        y_class = yhat.argmax(axis=-1)
        labels = self.labels

        rospy.loginfo('%s (%.2f%%) : ImagePrep time (s) : %f GPU time (s) : %f', labels[y_class],
                      yhat[y_class]*100, dt1.to_sec(), dt2.to_sec())

        self.current_light = TrafficLight.UNKNOWN
        if (yhat[y_class] > 0.5):
            if y_class == 0:
                self.current_light = TrafficLight.GREEN
            elif y_class == 2:
                self.current_light = TrafficLight.RED
            elif y_class == 3:
                self.current_light = TrafficLight.YELLOW

        return self.current_light
