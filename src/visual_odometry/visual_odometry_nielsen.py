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
        self.odom_pub = rospy.Publisher('/doma_planner/odom_nielsen', Odometry, queue_size=10)

        self.rate = rospy.Rate(10)
        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)

        self.main()