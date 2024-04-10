#!/usr/bin/env python

import numpy as np
import cv2
import rospy
import functools
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

class VisualOdometry:
    def __init__(self):
        rospy.init_node('doma_visual_odometry_node_nielsen', anonymous=True)
        self.bridge = CvBridge()
        self.odometry = Odometry()
        self.curr_image = None
        self.prev_image = None
        self.FLAN_INDEX_LSH = 6
        self.orb = cv2.ORB_create(3000)
        self.index_params = dict(algorithm=self.FLAN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.image_width = rospy.get_param('image_width')
        self.image_height = rospy.get_param('image_height')
        self.camera_matrix = rospy.get_param('camera_matrix')
        self.fx = self.camera_matrix['data'][0]
        self.fy = self.camera_matrix['data'][4]
        self.cx = self.camera_matrix['data'][2]
        self.cy = self.camera_matrix['data'][5]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.P = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])
        self.distortion_coefficients = rospy.get_param('distortion_coefficients')
        self.projection_matrix = rospy.get_param('projection_matrix')

        self.image_sub = rospy.Subscriber('/doma_planner/raw_image', Image, self.image_callback)
        self.odom_pub = rospy.Publisher('/doma_planner/odom_nielsen', Odometry, queue_size=10)

        self.rate = rospy.Rate(10)
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

        self.main()

    def shutdownhook(self):
        rospy.loginfo('Shutting down Visual Odometry Node...')
        self.ctrl_c = True

    @staticmethod
    def _form_transform(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def image_callback(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            # Change the image to grayscale
            self.curr_image = cv2.cvtColor(self.curr_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            rospy.logerr('Error in node doma_visual_odometry_node_nielsen: {}'.format(e))


    def get_matches(self):
        key_points1, descriptors1 = self.orb.detectAndCompute(self.prev_image, None)
        key_points2, descriptors2 = self.orb.detectAndCompute(self.curr_image, None)

        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        try:
            for m,n in matches:
                if m.distance < n.distance:
                    good_matches.append(m)

            q1 = np.float32([key_points1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([key_points2[m.trainIdx].pt for m in good_matches])

            return q1, q2
        
        except Exception as e:
            rospy.logerr('Error in node doma_visual_odometry_node_nielsen: {}'.format(e))
            return None, None
    
    def get_pose(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(E, q1, q2)
        return self._form_transform(R, t)
    
    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transform(R1, np.ndarray.flatten(t))
        T2 = self._form_transform(R2, np.ndarray.flatten(t))
        T3 = self._form_transform(R1, np.ndarray.flatten(-t))
        T4 = self._form_transform(R2, np.ndarray.flatten(-t))

        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)

        projectiions = [K @ T for T in transformations]

        np.set_printoptions(suppress=True)

        positives = []

        for P, T in zip(projectiions, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q1[2, :] > 0) + sum(Q2[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=1) / np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if max == 2:
            return R1, np.ndarray.flatten(-t)
        elif max == 3:
            return R2, np.ndarray.flatten(-t)
        elif max == 0:
            return R1, np.ndarray.flatten(t)
        elif max == 1:
            return R2, np.ndarray.flatten(t)
        
    def rot2quat(self, R):
        qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)

        return qw, qx, qy, qz

            
    def main(self):
        while not self.ctrl_c:
            try:
                if self.curr_image is not None and self.prev_image is not None:
                    q1, q2 = self.get_matches()
                    T = self.get_pose(q1, q2)
                    t = T[:3, 3]
                    R = T[:3, :3]
                    qw, qx, qy, qz = self.rot2quat(R)

                    self.odometry.header.stamp = rospy.Time.now()
                    self.odometry.header.frame_id = 'odom'
                    self.odometry.child_frame_id = 'base_link'
                    self.odometry.pose.pose.position = Point(t[0], t[1], t[2])
                    self.odometry.pose.pose.orientation = Quaternion(qx, qy, qz, qw)

                    self.odom_pub.publish(self.odometry)
                    self.prev_image = self.curr_image
                else:
                    self.prev_image = self.curr_image

                self.rate.sleep()
            except Exception as e:
                rospy.logerr('Error in node doma_visual_odometry_node_nielsen: {}'.format(e))

if __name__ == '__main__':
    vo = VisualOdometry()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down Visual Odometry Node...')
        cv2.destroyAllWindows()

    