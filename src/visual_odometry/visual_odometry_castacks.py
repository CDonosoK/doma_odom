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
        rospy.init_node('doma_visual_odometry_node_castacks', anonymous=True)
        self.bridge = CvBridge()
        self.odometry = Odometry()
        self.image = None
        self.prev_image = None

        self.image_width = rospy.get_param('image_width')
        self.image_height = rospy.get_param('image_height')
        self.camera_matrix = rospy.get_param('camera_matrix')
        self.fx = self.camera_matrix['data'][0]
        self.fy = self.camera_matrix['data'][4]
        self.cx = self.camera_matrix['data'][2]
        self.cy = self.camera_matrix['data'][5]
        self.distortion_coefficients = rospy.get_param('distortion_coefficients')
        self.projection_matrix = rospy.get_param('projection_matrix')

        self.image_sub = rospy.Subscriber('/doma_planner/raw_image', Image, self.image_callback)
        self.odom_pub = rospy.Publisher('/doma_planner/odom_castacks', Odometry, queue_size=10)

        self.rate = rospy.Rate(10)
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

        self.main()

    def shutdownhook(self):
        rospy.loginfo('Shutting down Visual Odometry Node...')
        self.ctrl_c = True

    def image_callback(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

    def rot2quat(self, R):
        rz, ry, rx = self.mat2euler(R)
        return self.euler2quat(rz, ry, rx)


    def euler2quat(self, z=0, y=0, x=0, isRadian=True):
        if not isRadian:
            z = math.radians(z)
            y = math.radians(y)
            x = math.radians(x)

        z = z / 2
        y = y / 2
        x = x / 2
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)

        return np.array([
            cx * cy * cz - sx * sy * sz,
            cx * sy * sz + cy * cz * sx,
            cx * cz * sy - sx * cy * sz,
            cx * cy * sz + sx * cz * sy
        ])
    
    def mat2euler(self, M, cy_thresh=None, seq='zyx'):
        M = np.asarray(M)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4.0
            except ValueError:
                cy_thresh = np.finfo(float).eps * 4.0

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        cy = math.sqrt(r33 * r33 + r23 * r23)

        if seq == 'zyx':
            if cy > cy_thresh:
                z = math.atan2(-r12, r11)
                y = math.atan2(r13, cy)
                x = math.atan2(-r23, r33)
            else:
                z = math.atan2(r21, r22)
                y = math.atan2(r13, cy)
                x = 0.0
        elif seq == 'xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)

            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi / 2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi / 2
                    x = math.atan2(-r12, -r13)
        else:
            rospy.logerr('Sequence not recognized. Use "zyx" or "xyz"')
        
        return x, y, z

    def main(self):
        while not self.ctrl_c:
            if self.image is not None:
                if self.prev_image is not None:
                    try:
                        orb_features = cv2.ORB_create(nfeatures=6000)

                        key_points1, descriptors1 = orb_features.detectAndCompute(self.prev_image, None)
                        key_points2, descriptors2 = orb_features.detectAndCompute(self.image, None)

                        brute_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                        matches = brute_matcher.match(descriptors1, descriptors2)
                        matches = sorted(matches, key=lambda x: x.distance)

                        img_matches = cv2.drawMatches(self.prev_image, key_points1, self.image, key_points2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                        points1 = np.float32([key_points1[m.queryIdx].pt for m in matches])
                        points2 = np.float32([key_points2[m.trainIdx].pt for m in matches])

                        E, mask = cv2.findEssentialMat(points1, points2, focal=self.fx, pp=(self.cx, self.cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        points1 = points1[mask.ravel() == 1]
                        points2 = points2[mask.ravel() == 1]
                        _, R, t, mask = cv2.recoverPose(E, points1, points2, focal=self.fx, pp=(self.cx, self.cy))

                        R = R.transpose()
                        t = -np.matmul(R, t)

                        current_image_keypoints = cv2.drawKeypoints(self.image, key_points2, None, color=(0, 255, 0), flags=0)

                        [tx, ty, tz] = t
                        qw, qx, qy, qz = self.rot2quat(R)

                        self.odometry.header.stamp = rospy.Time.now()
                        self.odometry.header.frame_id = 'odom'
                        self.odometry.child_frame_id = 'base_link'
                        self.odometry.pose.pose.position = Point(tx, ty, tz)
                        self.odometry.pose.pose.orientation = Quaternion(qx, qy, qz, qw)
                        self.odometry.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

                        self.odom_pub.publish(self.odometry)
                    except Exception as e:
                        rospy.logerr('Error in DOMA Visual Odometry: %s', e)
                

                self.prev_image = self.image

            self.rate.sleep()

if __name__ == '__main__':
    try:
        VisualOdometry()
    except rospy.ROSInterruptException:
        pass