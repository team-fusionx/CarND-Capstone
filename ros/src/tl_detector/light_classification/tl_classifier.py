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
    def __init__(self, use_image_clips):
        rospy.logwarn("TLClassifier __init__ begins")
        t_0 = rospy.Time.now()
        # Clear any prev Keras sessions to improve model load time
        rospy.logwarn("clear_session")

        # Parameter from tl_detector launch files to select site vs sim model
        self.use_image_clips = use_image_clips

        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        if self.use_image_clips is True:
            model_name = "ResNet50-UdacityReal_5runs-1.0"
        else:
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
        self.labels = np.array(['GREEN', 'NONE', 'RED', 'YELLOW'])
        self.resize_width = 224
        self.resize_height = 224
        self.last_clip_found = 0
        t_1 = rospy.Time.now()
        dt_1 = t_1 - t_0
        rospy.logwarn('TL Classifier init completed in %f s.',dt_1.to_sec())

    def get_classification(self, image, image_counter):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        labels = self.labels

        t0 = rospy.Time.now()

        # Switch classification method between site test (classify by multiple
        # image clips) vs simulator (classify by single full image)
        if self.use_image_clips is True:
            # Classify by multiple image clips
            # Expecting 800x600x3 images
            # first check that image is 800x600 - if not resize it.
            if image.shape[:2] != (600,800):
                rospy.loginfo("Resizing image from {} to {}".format(image.shape[:2][::-1], (800,600)))
                image = cv2.resize(image,(800,600))
            ###chop image up
            detect = False
            # left corner x co-ords to split 800 pixels into 5 sections of 224 pixels
            startx = [0,152,288,424,576]
            # store the maximum confidence for green, yellow and red in an array
            max_gyr = [0.0,0.0,0.0]
            #store the minimum confidence for finding nothing
            min_none = 1.0
            # pointers  between index in green yellow and red in gyr to yhat array
            gyr_to_color = [0,3,2]  # g,y,r to g,n,r,y
            # list to store details of results - not used yet
            foundinclip = []
            # use different search pattern based on which clip tl was identified previously 
            search_paths = [[0,1,2,3,4],[1,0,2,3,4],[2,1,3,0,4],[3,2,4,1,0],[4,3,2,1,0]]
            
            for i in search_paths[self.last_clip_found]:
                # run classification on a clip from the middle section of the image
                image_clip = image[188:412, startx[i]:startx[i]+224]
                np_image_data = np.asarray(image_clip)
                np_final = np.expand_dims(np_image_data, axis=0)
                np_final = resnet50.preprocess_input(np_final.astype('float64'))

                yhat = self.sess.run(self.y, feed_dict={self.x: np_final})

                yhat = yhat[0]
                y_class = yhat.argmax(axis=-1)

                # green
                if yhat[0] > max_gyr[0]:
                    max_gyr[0] = yhat[0]
                # red
                if yhat[2] > max_gyr[2]:
                    max_gyr[2] = yhat[2]
                # yellow
                if yhat[3] > max_gyr[1]:
                    max_gyr[1] = yhat[3]
                # none
                min_none = min(min_none, yhat[1])

                rospy.loginfo("Image {} Clip {}, {}:{:4.2f}%, {}:{:4.2f}%, {}:{:4.2f}%, {}:{:4.2f}% "
                              .format(image_counter, i, labels[0], yhat[0]*100.0, labels[3], yhat[3]*100.0, labels[2], yhat[2]*100.0, labels[1], yhat[1]*100.0))
                
                if y_class != 1:
                    detect = True
                    foundinclip.append((i, y_class, yhat[y_class]*100.0))
                    if yhat[y_class] > 0.6:
                        # fairly confident found a light so stop looking
                        self.last_clip_found = i
                        break
            
            dt2 = rospy.Time.now() - t0

            if detect is True:
                gyr_index = np.argmax(max_gyr)
                confidence = max_gyr[gyr_index]
                color_index = gyr_to_color[gyr_index]

            else:
                confidence = min_none  # use lowest confidence for none
                color_index = 1

            rospy.loginfo('%s (%.2f%%) | GPU time (s) : %f', labels[color_index],
                          confidence*100, dt2.to_sec())

        else:
            # Classify by single full image
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

            confidence = yhat[y_class]
            color_index = y_class


            rospy.loginfo('%s (%.2f%%) : ImagePrep time (s) : %f GPU time (s) : %f', labels[y_class],
                          yhat[y_class]*100, dt1.to_sec(), dt2.to_sec())

        self.current_light = TrafficLight.UNKNOWN
        if (confidence > 0.6):
            if color_index == 0:
                self.current_light = TrafficLight.GREEN
            elif color_index == 2:
                self.current_light = TrafficLight.RED
            elif color_index == 3:
                self.current_light = TrafficLight.YELLOW

        return self.current_light
