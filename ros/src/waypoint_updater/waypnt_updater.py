#!/usr/bin/env python
# File renamed to waypnt_updater.py because original name
# waypoint_updater.py which matches directory and package name
# prevented loading from waypoint_updater.cfg

import rospy
import math
import copy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
from std_msgs.msg import String, Int32
from dynamic_reconfigure.server import Server
from waypoint_updater.cfg import DynReconfConfig


'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.  As mentioned in the doc, you should ideally first implement
a version which does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of
traffic lights too.

Please note that our simulator also provides the exact location of traffic
lights and their current status in `/vehicle/traffic_lights` message. You
can use this message to build this node as well as to verify your TL
classifier.

'''


def get_accel_distance(Vi, Vf, A):
    return math.fabs((Vf**2 - Vi**2)/(2.0 * A))


def get_accel_time(S, Vi, Vf):
    return math.fabs((2.0 * S / (Vi + Vf)))


class JMT(object):
    def __init__(self, start, end, T):
        """
        Calculates Jerk Minimizing Trajectory for start, end and T.
        start and end include
        [displacement, velocity, acceleration]
        """
        self.start = start
        self.end = end
        self.final_displacement = end[0]
        self.T = T

        a_0, a_1, a_2 = start[0], start[1], start[2] / 2.0
        c_0 = a_0 + a_1 * T + a_2 * T**2
        c_1 = a_1 + 2 * a_2 * T
        c_2 = 2 * a_2

        A = np.array([
                     [T**3,   T**4,    T**5],
                     [3*T**2, 4*T**3,  5*T**4],
                     [6*T,   12*T**2, 20*T**3],
                     ])

        B = np.array([
                     end[0] - c_0,
                     end[1] - c_1,
                     end[2] - c_2
                     ])
        a_3_4_5 = np.linalg.solve(A, B)
        self.coeffs = np.concatenate([np.array([a_0, a_1, a_2]), a_3_4_5])

    # def JMTD_at(self, displacement, coeffs, t0, tmax, deq_wpt_ptr):
    def JMTD_at(self, displacement, t0, tmax):
        # find JMT descriptors at displacement
        s_last = 0.0
        t_found = False
        t_inc = 0.01
        iterations = 0

        for t_cnt in range(int((tmax-t0)/t_inc) + 100):
            iterations += 1
            t = t0 + t_cnt * t_inc
            s = self.get_s_at(t)
            if s > displacement:
                t = t - (1 - (s - displacement) / (s - s_last)) * t_inc
                t_found = True
                break
            # end if
            s_last = s
        # end for
        if t_found is False:
            rospy.loginfo("waypoint_updater:JMTD_at Ran out of bounds without "
                          "finding target displacement")
            return None

        s = self.get_s_at(t)
        delta_s = (displacement - s)
        if delta_s > 0.0:
            searchdir = 1.0
        else:
            searchdir = -1.0

        t_smallinc = searchdir * 0.005
        while delta_s * searchdir > 0.0:
            iterations += 1
            t += t_smallinc
            delta_s = displacement - self.get_s_at(t)

        rospy.loginfo("delta_s = {}, t= {}, iterations = {}".format(
            delta_s, t, iterations))
        if delta_s > 0.1:
            rospy.loginfo("waypoint_updater:JMTD_at need to refine algo,"
                          " delta_s is {}".format(delta_s))

        details = JMTDetails(self.get_s_at(t), self.get_v_at(t),
                             self.get_a_at(t), self.get_j_at(t), t)

        rospy.loginfo("waypoint_updater:JMTD_at displacement {} found "
                      "s,v,a,j,t = {}".format(displacement, details))

        return details

    def get_s_at(self, t):
        return self.coeffs[0] + self.coeffs[1] * t + self.coeffs[2] * t**2 +\
            self.coeffs[3] * t**3 + self.coeffs[4] * t**4 + self.coeffs[5] *\
            t**5

    def get_v_at(self, t):
        return self.coeffs[1] + 2.0 * self.coeffs[2]*t + 3.0 *\
            self.coeffs[3] * t**2 + 4.0 * self.coeffs[4] *\
            t**3 + 5.0 * self.coeffs[5] * t**4

    def get_a_at(self, t):
        return 2.0 * self.coeffs[2] + 6.0 * self.coeffs[3] * t + 12.0 *\
            self.coeffs[4] * t**2 + 20.0 * self.coeffs[5] * t**3

    def get_j_at(self, t):
        return 6.0 * self.coeffs[3] + 24.0 * self.coeffs[4] * t + 60.0 *\
            self.coeffs[5] * t**2


class JMTDetails(object):
    def __init__(self, S, V, A, J, t):
        self.S = S
        self.V = V
        self.A = A
        self.J = J
        self.time = t

    def set_VAJt(self, V, A, J, time):
        self.V = V
        self.A = A
        self.J = J
        self.time = time

    def __repr__(self):
        return "%5.3f, %2.4f, %2.4f, %2.4f, %2.3f" % (self.S, self.V, self.A,
                                                      self.J, self.time)


class JMTD_waypoint(object):
    # def __init__(self, xval, yval, zval, max_v, ptr_id, s):
    def __init__(self, waypoint, ptr_id, s):
        # self.position = pose(xval, yval, zval)
        self.waypoint = waypoint
        self.max_v = self.get_v()  # max_v is read only
        self.set_v(0.0)
        self.JMTD = JMTDetails(s, 0.0, 0.0, 0.0, 0.0)
        self.ptr_id = ptr_id
        self.state = None
        self.JMT_ptr = -1  # points to JMT object

    def set_v(self, v):
        # put a check for max_v here
        self.waypoint.twist.twist.linear.x = min(v, self.max_v)
        self.JMTD.V = v

    def get_v(self):
        return self.waypoint.twist.twist.linear.x

    def get_position(self):
        return self.waypoint.pose.pose.position

    def get_x(self):
        return self.waypoint.pose.pose.position.x

    def get_y(self):
        return self.waypoint.pose.pose.position.y

    def get_z(self):
        return self.waypoint.pose.pose.position.z

    def get_s(self):
        return self.JMTD.S

    def get_maxV(self):
        return self.max_v


class WaypointUpdater(object):
    def __init__(self):
        self.dyn_vals_received = False
        self.waypoints = []
        self.pose = None
        self.velocity = None
        self.lights = None
        self.final_waypoints = []
        self.final_waypoints_start_ptr = 0
        self.back_search = False
        self.last_search_distance = None
        self.last_search_time = None
        self.next_tl_wp = None
        self.update_rate = 10
        self.max_velocity = 0.0
        self.max_accel = 5.0
        self.max_jerk = 5.0
        self.default_velocity = 10.7
        self.lookahead_wps = 50  # 200 is too many
        self.subs = {}
        self.pubs = {}
        self.dyn_reconf_srv = None
        self.max_s = 0.0  # length of track
        self.JMT_List = []
        # target max acceleration/braking force - dynamically adjustable
        self.default_accel = 1.5
        self.state = 'stopped'  # for now only use to see if stopped or moving

        rospy.init_node('waypoint_updater')

        self.subs['/base_waypoints'] = \
            rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.subs['/current_pose'] = \
            rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        self.subs['/current_velocity'] = \
            rospy.Subscriber('/current_velocity', TwistStamped,
                             self.velocity_cb)

        self.subs['/traffic_waypoint'] = \
            rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint

        self.pubs['/final_waypoints'] = rospy.Publisher('/final_waypoints',
                                                        Lane, queue_size=1)

        # self.dyn_reconf_srv = Server(DynReconfConfig, self.dyn_vars_cb)
        # move to waypoints_cb so no collision - seems like a bad idea but
        # convenient for now

        self.loop()

    def loop(self):
        self.rate = rospy.Rate(self.update_rate)
        # wait for waypoints and pose to be loaded before trying to update
        # waypoints
        while not self.waypoints:
            self.rate.sleep()
        while not self.pose:
            self.rate.sleep()
        while not rospy.is_shutdown():

            if self.waypoints:
                self.send_waypoints()
            self.rate.sleep()

    # adjust dynamic variables
    def dyn_vars_cb(self, config, level):
        self.dyn_vals_received = True
        if self.update_rate:
            old_update_rate = self.update_rate
            old_default_velocity = self.default_velocity
            old_default_accel = self.default_accel
            old_lookahead_wps = self.lookahead_wps
            old_test_stoplight_wp = self.next_tl_wp
        # end if

        rospy.loginfo("Received dynamic parameters {} with level: {}"
                      .format(config, level))

        if old_update_rate != config['dyn_update_rate']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting update Rate "
                          "from {} to {}".format(old_update_rate,
                                                 config['dyn_update_rate']))
            self.update_rate = config['dyn_update_rate']
            # need to switch the delay
            self.rate = rospy.Rate(self.update_rate)
        # end if

        if old_default_velocity != config['dyn_default_velocity']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting default_"
                          "velocity from {} to {}"
                          .format(old_default_velocity,
                                  config['dyn_default_velocity']))

            if config['dyn_default_velocity'] > self.max_velocity:
                rospy.logwarn("waypoint_updater:dyn_vars_cb default_velocity "
                              "limited to max_velocity {}"
                              .format(self.max_velocity))
                self.default_velocity = self.max_velocity * 0.975
            else:
                self.default_velocity = config['dyn_default_velocity']
            # end if
        # end if

        if old_default_accel != config['dyn_default_accel']:
            rospy.logwarn("waypoint_updater:dyn_vars_cb Adjusting default_"
                          "accel from {} to {}"
                          .format(old_default_accel,
                                  config['dyn_default_accel']))
            self.default_accel = config['dyn_default_accel']
        # end if

        if old_lookahead_wps != config['dyn_lookahead_wps']:
            rospy.loginfo("waypoint_updater:dyn_vars_cb Adjusting lookahead_"
                          "wps from {} to {}"
                          .format(old_lookahead_wps,
                                  config['dyn_lookahead_wps']))
            self.lookahead_wps = config['dyn_lookahead_wps']
        # end if

        if old_test_stoplight_wp != config['dyn_test_stoplight_wp']:
            rospy.logwarn("waypoint_updater:dyn_vars_cb Adjusting next "
                          "stoplight from {} to {}"
                          .format(old_test_stoplight_wp,
                                  config['dyn_test_stoplight_wp']))
            self.next_tl_wp = config['dyn_test_stoplight_wp']
        # end if

        # we can also send adjusted values back
        return config

    def velocity_cb(self, twist_msg):
        # Check this is right
        self.velocity = twist_msg.twist.linear.x
        # TODO remove next line when verified correct
        rospy.loginfo("Velocity reported as {}".format(self.velocity))

    def pose_cb(self, pose_msg):
        # TODO refactor this to get position pose.pose.position
        self.pose = pose_msg.pose
        rospy.logdebug("waypoint_updater:pose_cb pose set to  %s", self.pose)

    # Load set of waypoints from /basewaypoints into self.waypoints
    # this should only happen once, so we unsubscribe at end
    def waypoints_cb(self, lane_msg):

        rospy.loginfo("waypoint_updater:waypoints_cb loading waypoints")
        if not self.waypoints:
            cntr = 0
            s = 0.0
            max_velocity = 0.0
            wpt = None  # just to stop linter complaining

            for lanemsg_wpt in lane_msg.waypoints:

                if cntr > 0 and wpt:
                    # won't come into here until after wpt loaded
                    # in previous loop
                    s += math.sqrt((wpt.get_x() - lanemsg_wpt.pose.pose.
                                    position.x)**2 +
                                   (wpt.get_y() - lanemsg_wpt.pose.pose.
                                    position.y)**2)

                wpt = JMTD_waypoint(lanemsg_wpt, cntr, s)
                self.waypoints.append(wpt)
                if max_velocity < wpt.get_v():
                    max_velocity = wpt.get_v()
                # end if
                cntr += 1

            rospy.loginfo("waypoints_cb {} waypoints loaded, last waypoint "
                          "ptr_id = {} at s= {}".
                          format(len(self.waypoints), self.waypoints[cntr-1].
                                 ptr_id, self.waypoints[cntr-1].get_s()))
            self.max_s = self.waypoints[cntr-1].get_s()
            # setting max velocity based on project requirements in
            # Waypoint Updater Node Revisited
            self.max_velocity = max_velocity
            rospy.loginfo("waypoint_updater:waypoints_cb max_velocity set to "
                          " {} based on max value in waypoints."
                          .format(self.max_velocity))
            # now max_velocity is known, set up dynamic reconfig
            if not self.dyn_reconf_srv:
                self.dyn_reconf_srv = Server(DynReconfConfig, self.dyn_vars_cb)
                rospy.loginfo("dynamic_parm server started")
            # end if
        else:
            rospy.logerr("waypoint_updater:waypoints_cb attempt to load "
                         "waypoints when we have already loaded %d waypoints",
                         len(self.waypoints))
        # end if else
        self.subs['/base_waypoints'].unregister()
        rospy.loginfo("Unregistered from /base_waypoints topic")

    # Receive a msg from /traffic_waypoint about the next stop line
    def traffic_cb(self, traffic_msg):
        if traffic_msg.data != self.next_tl_wp:
            self.next_tl_wp = traffic_msg.data
            rospy.loginfo("new /traffic_waypoint message received at wp: %d."
                          "while car is at wp %d", self.next_tl_wp,
                          self.final_waypoints_start_ptr)
        else:
            # just for debug to see what we're getting
            rospy.loginfo("same /traffic_waypoint message received.")

    # adjust the velocities in the /final_waypoint queue
    def set_waypoints_velocity(self):
        offset = 1  # offset in front of car to account for some latency
        safety_factor = 1.2  # additional space for slowing down
        tl_buffer = 3.0  # distance from tl to stop
        recalc = False
        t = 0.0

        # this can happen before we get a traffic_wp msg
        if not self.next_tl_wp:
            # set one up behind car - is this a problem if we reverse?
            # STUB in putting it at 759 to see if it works
            # next_tl_wp = self.final_waypoints_start_ptr - 1
            self.next_tl_wp = 759
        if self.next_tl_wp > self.final_waypoints_start_ptr:
            # TODO does not account for looping
            dist_to_tl = self.waypoints[self.next_tl_wp - 1].get_s() -\
                self.waypoints[self.final_waypoints_start_ptr + offset].get_s()
        else:
            dist_to_tl = 1000  # big number
        if self.velocity == 0.0 and\
                self.waypoints[self.final_waypoints_start_ptr].get_v() == 0.0:
            # we are stopped
            offset = 0
            stopping_distance = 0.0
            self.state = 'stopped'
        else:
            stopping_distance = get_accel_distance(
                self.waypoints[self.final_waypoints_start_ptr+offset].get_v(),
                0.0, self.default_accel) + tl_buffer
            self.state = 'moving'

        # handle case where we are stopped at lights and light is red
        if self.state == 'stopped' and dist_to_tl < tl_buffer:
            for ptr in range(self.final_waypoints_start_ptr, self.
                             final_waypoints_start_ptr + self.lookahead_wps):
                mod_ptr = ptr % len(self.waypoints)
                self.waypoints[mod_ptr].JMTD.set_VAJt(0.0, 0.0, 0.0, 0.0)
                self.waypoints[mod_ptr].set_v(0.0)

        elif dist_to_tl < stopping_distance * safety_factor:
            # slowdown or stop
            start_ptr = self.final_waypoints_start_ptr + offset
            #  end_ptr = self.next_tl_wp
            final_end_ptr = self.final_waypoints_start_ptr + self.lookahead_wps
            if dist_to_tl < tl_buffer:  # small buffer from stop line
                for ptr in range(start_ptr, final_end_ptr):
                    mod_ptr = ptr % len(self.waypoints)
                    self.waypoints[mod_ptr].set_v(0.0)
                    self.waypoints[mod_ptr].JMTD.set_VAJt(0.0, 0.0, 0.0, 0.0)
                # end for
            else:
                target_velocity = 0.0
                curpt = self.waypoints[start_ptr]
                jmt_ptr = self.setup_jmt(curpt, target_velocity)
                curpt.JMT_ptr = jmt_ptr
                JMT_instance = self.JMT_List[jmt_ptr]
                for ptr in range(start_ptr+1, final_end_ptr):

                    mod_ptr = ptr % len(self.waypoints)
                    wpt = self.waypoints[mod_ptr]

                    if wpt.JMTD.S > JMT_instance.final_displacement:
                        rospy.loginfo("Passed beyond S = {} at ptr_id = {}".
                                      format(JMT_instance.final_displacement,
                                             mod_ptr))
                        # assume that targets achieved
                        wpt.JMTD.set_VAJt(target_velocity, 0.0, 0.0, 0.0)
                    else:
                        jmt_point = JMT_instance.JMTD_at(
                            wpt.JMTD.S, t, JMT_instance.T*1.5)
                        if jmt_point is None:
                            rospy.loginfo("JMT_at returned None at ptr_id = {}"
                                          .format(mod_ptr))
                            break

                        if self.check_jmt_point(jmt_point, mod_ptr) is True:
                            recalc = True
                        t = jmt_point.time
                        wpt.JMTD.set_VAJt(jmt_point.V, jmt_point.A,
                                          jmt_point.J, jmt_point.time)
                        wpt.set_v(jmt_point.V)
                    # end if else
                # end for
            # end if else
        else:
            # TODO Handle startup and JMTSpeedup
            if self.waypoints[mod_ptr].get_v() >= self.default_velocity:
                start_ptr = self.final_waypoints_start_ptr + offset
                final_end_ptr = self.final_waypoints_start_ptr +\
                    self.lookahead_wps
                for ptr in range(start_ptr, final_end_ptr):
                    mod_ptr = ptr % len(self.waypoints)
                    self.waypoints[mod_ptr].set_v(self.default_velocity)
                    self.waypoints[mod_ptr].JMTD.\
                        set_VAJt(self.default_velocity, 0.0, 0.0, 0.0)
                offset = self.lookahead_wps  # this will bar entry to loop

            if self.state == 'stopped':
                velocity = 0.0
                start_ptr = self.final_waypoints_start_ptr
                final_end_ptr = self.final_waypoints_start_ptr +\
                    self.lookahead_wps
                offset = 0
                while velocity < 1.5 and offset < self.lookahead_wps:
                    disp = self.waypoints[start_ptr + offset].get_s() -\
                        self.waypoints[start_ptr].get_s()
                    velocity = max(0.5, math.sqrt(self.default_accel * disp *
                                                  2.0))
                    velocity = min(velocity, self.waypoints
                                   [start_ptr + offset].get_maxV())
                    self.waypoints[start_ptr + offset].set_v(velocity)
                    self.waypoints[start_ptr + offset].JMTD.set_VAJt(
                        velocity, self.default_accel, 0.0, 0.0)
                    offset += 1
                self.state = 'moving'
            # end if stopped

            start_ptr = self.final_waypoints_start_ptr + offset
            final_end_ptr = self.final_waypoints_start_ptr +\
                self.lookahead_wps
            curpt = self.waypoints[start_ptr]
            jmt_ptr = self.setup_jmt(curpt, self.default_velocity)
            curpt.JMT_ptr = jmt_ptr
            JMT_instance = self.JMT_List[jmt_ptr]
            for ptr in range(start_ptr+1, final_end_ptr):

                mod_ptr = ptr % len(self.waypoints)
                wpt = self.waypoints[mod_ptr]

                if wpt.get_s() > JMT_instance.final_displacement:
                    rospy.loginfo("Passed beyond S = {} at ptr_id = {}".
                                  format(JMT_instance.final_displacement,
                                         mod_ptr))
                    # assume that targets achieved
                    wpt.JMTD.set_VAJt(self.default_velocity, 0.0, 0.0, 0.0)
                    wpt.set_v(self.default_velocity)
                else:
                    jmt_point = JMT_instance.JMTD_at(
                        wpt.JMTD.S, t, JMT_instance.T*1.5)
                    if jmt_point is None:
                        rospy.loginfo("JMT_at returned None at ptr_id = {}"
                                      .format(mod_ptr))
                        break

                    if self.check_jmt_point(jmt_point, mod_ptr) is True:
                        recalc = True
                    t = jmt_point.time
                    wpt.JMTD.set_VAJt(jmt_point.V, jmt_point.A,
                                      jmt_point.J, jmt_point.time)
                    wpt.set_v(jmt_point.V)
                # end if else
            # end for
        # end if
        if recalc is True:
            rospy.logwarn("recalc is set to {} we should recalculate"
                          .format(recalc))

        rospy.loginfo("ptr_id, S, V, A, J, time")
        for wpt in self.waypoints[self.final_waypoints_start_ptr:
                                  self.final_waypoints_start_ptr +
                                  self.lookahead_wps]:
            rospy.loginfo("{}, {}".format(wpt.ptr_id, wpt.JMTD))

    def setup_jmt(self, curpt, target_velocity):

        a_dist = get_accel_distance(curpt.JMTD.V, target_velocity,
                                    self.default_accel)
        T = get_accel_time(a_dist, curpt.JMTD.V, target_velocity)

        rospy.loginfo("Setup JMT to accelerate to {} in dist {} m in time {} s"
                      .format(target_velocity, a_dist, T))

        start = [curpt.JMTD.S, curpt.JMTD.V, curpt.JMTD.A]
        end = [curpt.JMTD.S + a_dist, target_velocity, 0.0]
        jmt = JMT(start, end, T)
        self.JMT_List.append(jmt)
        jmt_ptr = len(self.JMT_List)

        return jmt_ptr-1

    def check_jmt_point(self, jmt_pnt, ptr_id):
        # check if JMT values at point exceed desired values
        recalc = False
        if jmt_pnt.A > self.max_accel:
            rospy.loginfo("A of {} exceeds max value of {} "
                          "at ptr = {}".format
                          (jmt_pnt.A, self.max_accel, ptr_id))
            recalc = True
        if jmt_pnt.J > self.max_jerk:
            rospy.loginfo("J of {} exceeds max value of {} "
                          "at ptr = {}".format
                          (jmt_pnt.J, self.max_jerk, ptr_id))
            recalc = True
        if jmt_pnt.V > self.max_velocity:
            rospy.loginfo("V of {} exceeds max value of {} "
                          "at ptr = {}".format
                          (jmt_pnt.V, self.max_velocity, ptr_id))
            recalc = True
        return recalc

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
        # We will implement it later
        pass

    def send_waypoints(self):
        # generates the list of LOOKAHEAD_WPS waypoints based on car location
        # for now assume waypoints form a loop - may not be the case

        self.final_waypoints_start_ptr = self.closest_waypoint()
        # end_wps_ptr = (self.final_waypoints_start_ptr +
        #                self.lookahead_wps) % len(self.waypoints)
        # rospy.loginfo("waypoint_updater:send_waypoints start_wps_ptr = %d,"
        #               " end_wps_ptr = %d", self.final_waypoints_start_ptr,
        #               end_wps_ptr)
        # if end_wps_ptr > self.final_waypoints_start_ptr:
        #     for w_p in self.waypoints[self.final_waypoints_start_ptr:
        #                               end_wps_ptr]:
        #         new_wps_list.append(w_p)
        #     # end of for
        # else:
        #     for w_p in self.waypoints[self.final_waypoints_start_ptr:]:
        #         new_wps_list.append(w_p)
        #     # end of for
        #     for w_p in self.waypoints[:end_wps_ptr]:
        #         new_wps_list.append(w_p)
        #     # end of for
        # # end of if

        self.set_waypoints_velocity()
        lane = Lane()
        waypoints = []
        for wpt in self.waypoints[self.final_waypoints_start_ptr:
                                  self.final_waypoints_start_ptr +
                                  self.lookahead_wps]:
            waypoints.append(wpt.waypoint)

        lane.waypoints = list(waypoints)
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        self.pubs['/final_waypoints'].publish(lane)

    def closest_waypoint(self):
        # TODO - use local search first of final_waypoints sent out last
        # iteration
        def distance_lambda(a, b): return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2)
        # TODO: move away from using final waypoint, just use waypoints
        # since we have saved original v info within the structure
        if self.final_waypoints:
            dist = distance_lambda(self.final_waypoints[0].get_position(),
                                   self.pose.position)
            for i in range(1, len(self.final_waypoints)):
                tmpdist = distance_lambda(self.final_waypoints[i].
                                          get_position(),
                                          self.pose.position)
                if tmpdist < dist:
                    dist = tmpdist
                else:
                    # distance is starting to get larger so look at
                    # last position
                    if (i == 1):
                        # we're closest to original waypoint, but what if
                        # we're going backwards - loop backwards to make sure
                        # a point further back  isn't closest
                        for j in range(self.final_waypoints_start_ptr-1,
                                       self.final_waypoints_start_ptr -
                                       len(self.final_waypoints),
                                       -1):
                            tmpdist = distance_lambda(
                                self.waypoints[j % len(self.waypoints)].
                                get_position(),
                                self.pose.position)
                            if tmpdist < dist:
                                dist = tmpdist
                                self.back_search = True
                            else:
                                if abs(dist-self.last_search_distance) < 5.0:
                                    self.last_search_distance = dist
                                    return ((j+1) % len(self.waypoints))
                                else:
                                    break
                            # end if else
                        # end for
                    # end if

                    if abs(dist-self.last_search_distance) < 5.0:
                        self.last_search_distance = dist
                        return ((self.final_waypoints_start_ptr + i - 1) %
                                len(self.waypoints))
                    # end if
                # end if else
            # end for - fall out no closest match that looks acceptable
            rospy.logwarn("waypoint_updater:closest_waypoint local search not"
                          "satisfied - run full search")
        # end if

        dist = 1000000  # maybe should use max
        closest = 0
        for i in range(len(self.waypoints)):
            tmpdist = distance_lambda(self.waypoints[i].get_position(),
                                      self.pose.position)
            if tmpdist < dist:
                closest = i
                dist = tmpdist
            # end of if
        # end of for
        self.last_search_distance = dist
        return closest
        # Note: the first waypoint is closest to the car, not necessarily in
        # front of it.  Waypoint follower is responsible for finding this

    # Todo need to adjust for looping
    def distance(self, wp1, wp2):
        dist = 0

        def distance_lambda(a, b): return math.sqrt((a.x-b.x)**2 +
                                                    (a.y-b.y)**2 +
                                                    (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += distance_lambda(self.waypoints[wp1].get_position(),
                                    self.waypoints[i].get_position())
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
