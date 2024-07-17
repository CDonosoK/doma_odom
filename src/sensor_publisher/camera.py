#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

'''
Class to publish the frontal camera view on the Image topic /frontal_camera

Parameters:
    camera_pub: publisher for the frontal image
    bridge: CvBridge to use cv2 image on ROS
    rate: rate of the node.
    ctrl_c: boolean to stop the node.
'''

class FrontalCamera():
    def __init__(self):
        rospy.init_node('doma_cammera_node', anonymous=True)
        self.camera_pub = rospy.Publisher('/doma_planner/raw_image', Image, queue_size=1)
        self.bridge = CvBridge()
        self.ctrl_c = False
        self.rate = rospy.Rate(30)


        rospy.on_shutdown(self.shutdownhook)

        self.run()

    def get_list_cameras(self):
        '''
        Function to get the list of cameras available on the computer
        '''
        for camera_index in range(10):
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Camera {camera_index} is not available")
            else:
                print(f"Camera {camera_index} is available")
                cap.release()

    def shutdownhook(self):
        '''
        Function to stop the node.
        '''
        self.ctrl_c = True

    def run(self):
        '''
        Function to open the camera with CV2 and export to ROS using the cvbridge
        '''
        cap = cv2.VideoCapture(-1)#6
        if not cap.isOpened():
            rospy.logerr("Error opening video stream")
            return
        
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if ret:
                try:
                    # Convierte el marco de OpenCV a un mensaje de imagen ROS
                    image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.camera_pub.publish(image_msg)
                except CvBridgeError as e:
                    rospy.logerr(e)
            self.rate.sleep()

        cap.release()

if __name__ == '__main__':
    try:
        frontal_camera = FrontalCamera()
    except rospy.ROSInterruptException:
        pass
