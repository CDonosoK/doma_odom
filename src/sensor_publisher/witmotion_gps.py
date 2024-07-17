#!/usr/bin/env python

import serial, rospy
from sensor_msgs.msg import NavSatFix

class GpsPublisher:
    def __init__(self) -> None:
        rospy.init_node("gps_node", anonymous=True)

        self.lat = 0
        self.lon = 0

        self.gps_publisher = rospy.Publisher('/doma_odom/gps', NavSatFix, queue_size=10)

        self.rate = rospy.Rate(10)
        self.usb = serial.Serial('/dev/ttyUSB0', 115200)

        self.ctrl_c = False

        rospy.on_shutdown(self.shutdownhook)
        
        self.publishData()


    def shutdownhook(self):
        self.ctrl_c = True

    def get_gps(self):
        data = self.usb.readline()
        data = data.decode('utf-8').split(',')
        print(data)
        
        try:
            if data[0] == '$GNGGA':
                satelital = int(data[7])
                self.alt = float(data[9])
                if data[3] == 'S':
                    datos = str(data[2])
                    deg = int(datos[0:2])
                    min = float(datos[2:])
                    self.lat = float("-"+str(deg + min/60))
                else:
                    datos = str(data[2])
                    deg = int(datos[0:2])
                    min = float(datos[2:])
                    self.lat = float(deg + min/60)
                if data[5] == 'W':
                    datos = str(data[4])
                    deg = int(datos[0:3])
                    min = float(datos[3:])
                    self.lon = float("-"+str(deg + min/60))
                else:
                    datos = str(data[4])
                    deg = int(datos[0:3])
                    min = float(datos[3:])
                    self.lon = float(deg + min/60)

                gps_data = NavSatFix()
                gps_data.latitude = self.lat
                gps_data.longitude = self.lon
                gps_data.altitude = self.alt
                gps_data.status.status = satelital
                gps_data.header.stamp = rospy.Time.now()
                gps_data.header.frame_id = "odom"

                self.gps_publisher.publish(gps_data)
                #rospy.loginfo("Satelites: %d, Lat: %f, Lon: %f, Alt: %f", satelital, self.lat, self.lon, self.alt)
        except Exception as e:
            pass

    def publishData(self):
        '''
        Function to publish the data
        '''
        while not self.ctrl_c:
            self.get_gps()
            self.rate.sleep()

if __name__ == "__main__":
    try:
        point_publisher = GpsPublisher()
    except rospy.ROSInterruptException:
        pass