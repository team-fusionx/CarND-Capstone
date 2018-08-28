#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import time
import numpy as np
import collections
from copy import deepcopy
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3 # Means we need 4 consecutives
# Although Carla is feeding at 30 Hz, we can handle at < 20 Hz
UPDATE_RATE = 20
CLASSIFY_BY_GROUND_TRUTH = False
tl_decoder = ['RED','YELLOW','GREEN','','UNKNOWN']


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        #Variable Definitions
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.image_counter = 0
        self.last_img_sum = 0
        # List of positions that correspond to the line to stop in front of for a given intersection
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_waypoints = []

        self.waypoints_2d = []  # store x,y pairs for use in KDTree
        self.waypoint_tree = None  # store KDTree in here

        self.bridge = CvBridge()

        # option to save classification images to images subfolder
        self.save_images = rospy.get_param('~save_images', False)
        self.image_q = collections.deque()

        if CLASSIFY_BY_GROUND_TRUTH:
            self.light_classifier = None
        else:
            use_image_clips = rospy.get_param('~use_image_clips', False)
            use_image_array = rospy.get_param('~use_image_array', False)
            self.light_classifier = TLClassifier(use_image_clips, use_image_array)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.RED
        self.last_state = TrafficLight.RED
        self.previous_light_state = TrafficLight.RED

        self.busy = False

        self.last_wp = -1
        self.state_count = STATE_COUNT_THRESHOLD
        self.L_update = False
        self.light_change_to_red_or_yellow = False

        self.current_velocity = None
        self.stopped_vel_thresh = 0.3  # m/s

        rospy.logdebug('Red: %s', TrafficLight.RED)
        rospy.logdebug('Yellow: %s', TrafficLight.YELLOW)
        rospy.logdebug('Green: %s', TrafficLight.GREEN)
        rospy.logdebug('Unknown: %s', TrafficLight.UNKNOWN)
        
        '''
        tl_detection node subscribes to:
        /base_waypoints provides the complete list of waypoints for the course.
        /current_pose can be used used to determine the vehicle's location.
        /image_color which provides an image stream from the car's camera. 
             These images are used to determine the color of upcoming traffic lights.
        /vehicle/traffic_lights provides the (x, y, z) coordinates of all traffic lights.
        '''

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 
        3D map space and helps you acquire an accurate ground truth data source for 
        the traffic light classifier by sending the current color state of all traffic 
        lights in the simulator. When testing on the vehicle, the color state will not 
        be available. You'll need to rely on the position of the light and 
        the camera image to predict it.
        '''
       
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        # Subscribe to /current_velocity for RED->UNKNOWN state validation
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.current_velocity_cb, queue_size=1)

        self.rate = rospy.Rate(UPDATE_RATE)
        rospy.logwarn('TL Detector init complete.')
        self.loop()

    def loop(self):
        loopcntr = 0
        while not rospy.is_shutdown():
            loopcntr = loopcntr + 1
            if len(self.image_q) > 0 and self.pose and self.waypoint_tree:

                """ Added the following to confirm classifier light state 
                    - to rule out any latency issue
                """	
                while len(self.image_q) > 0:

                    light_wp, gt_state, tl_state = self.process_traffic_lights()
            
                    if self.light_classifier is not None:
                        state = tl_state
                    else:		
                        state = gt_state
                    '''
                    Publish upcoming red lights at camera frequency.
                    Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                    of times till we start using it. Otherwise the previous stable state is
                    used.
                    '''

                    rospy.loginfo('TL: state_count={}, self.state={}, state={}, loop counter = {}'
                                .format(self.state_count, self.state, state, loopcntr))

                    if self.state != state:
                        if (self.state == TrafficLight.YELLOW 
                            and state == TrafficLight.RED):
                                rospy.logwarn('Approaching RED Light state - Required to stop')     
                        else:
                            self.state_count = 0
                        self.state = state
                        self.L_update = True 
                    elif self.state_count >= STATE_COUNT_THRESHOLD:
                        '''
                        Condition to update wp only when light changing to red or yellow
                        '''

                        if ((self.last_state == TrafficLight.GREEN 
                            or self.last_state == TrafficLight.YELLOW
                            or self.last_state == TrafficLight.UNKNOWN)
                            and state == TrafficLight.YELLOW):
                                self.light_change_to_red_or_yellow = True
                        elif state == TrafficLight.RED:
                                self.light_change_to_red_or_yellow = True
                        else:		
                                self.light_change_to_red_or_yellow = False
                        
                        #Update last state and waypoint        
                        self.last_state = self.state

                        if self.light_change_to_red_or_yellow:
                            light_wp = light_wp
                        else:
                            light_wp = -1

                        self.last_wp = light_wp
                        self.upcoming_red_light_pub.publish(Int32(light_wp))

                        if self.L_update:
                            self.L_update = False
                            if self.light_classifier is not None:
                                rospy.logwarn('Upcoming GT Light state: %s',tl_decoder[gt_state])     
                                rospy.logwarn('Upcoming Classifier Light state: %s',tl_decoder[state])
                            else:
                                rospy.logwarn('Upcoming GT Light state: %s',tl_decoder[state])     
                                rospy.logwarn('Upcoming Classifier Light state: %s',tl_decoder[tl_state])			
                            rospy.logwarn('Upcoming Stop line: %f',light_wp)
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                    self.state_count += 1
                else:
                    if len(self.image_q) == 0:
                        rospy.loginfo("Skipping tl_detector loop because no images in queue")
                    else:
                        rospy.loginfo("Skipping tl_detector loop because pose or waypoint_tree are False")
            self.rate.sleep()

    def pose_cb(self, msg):
        """Callback fuction for vehicle's location."""
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """Callback fuction for list of all waypoints."""
        t_waypoints = waypoints

        # Build KD tree for closest waypoint search
        t_waypoints_2d = []
        for wp in t_waypoints.waypoints:
            t_waypoints_2d.append([wp.pose.pose.position.x,
                                   wp.pose.pose.position.y])
        t_waypoint_tree = KDTree(t_waypoints_2d)

        self.waypoints = t_waypoints
        self.waypoints_2d = t_waypoints_2d
        self.waypoint_tree = t_waypoint_tree

        # Build list of closest waypoint to each stop line position
        t_stop_line_waypoints = []
        for pts in self.stop_line_positions:
            sl_wp = self.get_closest_waypoint(pts[0], pts[1])
            t_stop_line_waypoints.append(sl_wp)

        self.stop_line_waypoints = t_stop_line_waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index 
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        image_sum = np.sum(msg)

        if image_sum != self.last_img_sum:
            # image counter not definitive, because in a callback without lock but gives idea for tracking       
            self.image_counter = self.image_counter + 1
            dequesize = len(self.image_q)
            if dequesize > 3:
                num = self.image_q.popleft()['num']
                rospy.loginfo("dropped image {} - deque of {} is too deep".format(num, dequesize))           
            self.image_q.append({'num': self.image_counter, 'image': deepcopy(msg)})
            self.last_img_sum = image_sum

            ########
            # save image stream to files
            if self.save_images is True:
                image_name = 'images/image_'+ str(self.image_counter) + '.png'
                cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imwrite(image_name, cv2_img)
            ########

        else:
            rospy.loginfo("Duplicate image received in consecutive image_cb")


    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x, y (float64): x, y position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoint_tree.query([x, y], 1)[1]

    def get_light_state(self):
        """Determines the current color of the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if len(self.image_q) == 0:
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # rospy.loginfo("Begin xfer to imgmsg")
        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        image_dict = self.image_q.popleft()
        image = image_dict['image'] 
        image_num = image_dict['num'] 
        cv_image = self.bridge.imgmsg_to_cv2(image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image, image_num)

    def process_traffic_lights(self):
        """Finds closest visible traffic light/ground truth data, if one exists, and determines its
            stop line position and state of the traffic light

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            int: ID of traffic light color identified by TL classifier - if enabled, else ID of UNKNOWN 

        """
        light = None
        car_position = 0

        # List of positions that correspond to the line to stop in front of for a given intersection
        #stop_line_positions = self.config['stop_line_positions']
	
        ntl_wp = -1
        gt_ntl_state =TrafficLight.UNKNOWN
        ntl_state =TrafficLight.UNKNOWN
        classified_tl_state =TrafficLight.UNKNOWN  	
        if self.light_classifier is not None: 
            if not self.busy:
                self.busy = True
                classified_tl_state = self.get_light_state()

                '''
                Update next Traffic Light state only for Valid Light state transitions		
                # State 0 - RED -> GREEN/RED or UNKNOWN if already stopped
                # State 1 - YELLOW -> RED/YELLOW/UNKNOWN
                # State 2 - GREEN -> YELLOW/GREEN/RED(upcoming signal was already RED)/UNKNOWN 
                # State 4 - Unknown -> RED/YELLOW/GREEN
                Assign next Traffic light state = previous state -
                   if there is no change in classification or 
                   if the predicted state doesnt meet the expected state			
                '''

                if ((self.previous_light_state == TrafficLight.UNKNOWN) 
                     or ((self.previous_light_state == TrafficLight.GREEN) 
                          and (classified_tl_state == TrafficLight.YELLOW)) 
                     or ((self.previous_light_state == TrafficLight.GREEN) 
                          and (classified_tl_state == TrafficLight.RED)) 
                     or ((self.previous_light_state == TrafficLight.GREEN) 
                          and (classified_tl_state == TrafficLight.UNKNOWN)) 
                     or ((self.previous_light_state == TrafficLight.YELLOW) 
                          and (classified_tl_state == TrafficLight.RED)) 
                     or ((self.previous_light_state == TrafficLight.YELLOW) 
                          and (classified_tl_state == TrafficLight.UNKNOWN)) 
                     or ((self.previous_light_state == TrafficLight.RED) 
                          and (classified_tl_state == TrafficLight.GREEN)) 
                     or ((self.previous_light_state == TrafficLight.RED)
                          and (classified_tl_state == TrafficLight.UNKNOWN)
                          and (self.current_velocity < self.stopped_vel_thresh))
                   ):
                    ntl_state= classified_tl_state
                    self.previous_light_state = ntl_state
                else:
                    ntl_state = self.previous_light_state

                self.busy = False
            else:
                rospy.loginfo("Skipping process traffic lights loop because busy")
                ntl_state = self.previous_light_state


        if self.pose and self.waypoint_tree:
            car_position = self.get_closest_waypoint(self.pose.pose.position.x,
                                                     self.pose.pose.position.y)

        # State = 0 : Red
        if ((ntl_state != 4) or (self.light_classifier is None)):
            if car_position and self.waypoint_tree:
                for tl in self.lights:
                    nearest_waypoint = self.get_closest_waypoint(
                                                    tl.pose.pose.position.x,
                                                    tl.pose.pose.position.y)

                    if nearest_waypoint > car_position:
                        ntl_wp = nearest_waypoint
                        gt_ntl_state = tl.state
                        break

        stop_line = -1 
        stop_distance = 10000
        dx = ntl_wp - car_position
        if dx>=0 and dx < 1000:
        #stop line nearest to the nearest light
            for position in self.stop_line_waypoints:
                if position < ntl_wp:
                    if ((ntl_wp - position) < stop_distance):
                        stop_distance = (ntl_wp - position)	
                        stop_line = position

        if stop_distance < 1000:
            return stop_line,gt_ntl_state, ntl_state 

        else:
            return -1, TrafficLight.UNKNOWN, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
