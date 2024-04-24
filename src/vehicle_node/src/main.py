#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import copy
import bisect
import matplotlib.cm as cm
import matplotlib.animation as animation

from IPython.display import HTML
from utils import *
from environment import Environments


import tf
import rospkg
import rospy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray

class Simulation(object):
    def __init__(self, dt=0.1):
        
        rospy.init_node('simulation')
        
        self.set_subscriber()
        self.set_publisher()
        self.initialize()
        
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            
            self.run()
            r.sleep()
            
    
    def initialize(self):
        self.pause = False
        course_idx = 0 # 0 : 좌회전 / 1 : 직진 / 2 : 우회전
        self.env = Environments(course_idx)

        
    def run(self):
        
        if self.pause:
            pass
        else:
            self.env.run()
            self.pub_track()
            self.pub_map()
            
            self.delete_agent()            
            
            self.env.respawn()
     
    def delete_agent(self):
        dl_list = self.env.delete_agent()
        
        self.pub_track(dl_list)
        
        for id_ in dl_list:            
            self.env.vehicles.pop(id_, None)
            self.env.int_pt_list.pop(id_, None) 
        
            
    def callback_plot(self, data):
            
        if data.linear.x>0 and data.angular.z>0: #u
            self.pause = True 
        else:
            self.pause = False
            
    def pub_track(self, delete_agent_list=[]):
        
        Objects = MarkerArray()
        Objects_v = MarkerArray()
        Objects_brake = MarkerArray()
        Objects_accel = MarkerArray()
        
        Texts = MarkerArray()
        
        
        for id_ in self.env.vehicles.keys():
            q = tf.transformations.quaternion_from_euler(0, 0, self.env.vehicles[id_].h)
            
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.id = id_
            marker.type = Marker.CUBE
            marker.pose.position.x = self.env.vehicles[id_].x 
            marker.pose.position.y = self.env.vehicles[id_].y
            marker.pose.position.z = 1.5
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3] 
            marker.scale.x = 5
            marker.scale.y = 2.5
            marker.scale.z = 3
            marker.color = ColorRGBA(0,0,1,1)

            if id_ in delete_agent_list:
                marker.action = Marker.DELETE
                
            Objects.markers.append(marker)
            
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.id = id_
            marker.type = Marker.CYLINDER
            marker.pose.position.x = self.env.vehicles[id_].x + 1*-np.sin(self.env.vehicles[id_].h)
            marker.pose.position.y = self.env.vehicles[id_].y + 1*np.cos(self.env.vehicles[id_].h)
            marker.pose.position.z = 8 + 1/2*self.env.vehicles[id_].v/5
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = self.env.vehicles[id_].v/5
            marker.color = ColorRGBA(0,1,0,1)  
            if id_ in delete_agent_list:
                marker.action = Marker.DELETE
                
                
            Objects_v.markers.append(marker)
            
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.id = id_
            marker.type = Marker.CYLINDER
            marker.pose.position.x = self.env.vehicles[id_].x 
            marker.pose.position.y = self.env.vehicles[id_].y
            marker.pose.position.z = 8 + 1/2*np.clip(self.env.vehicles[id_].ax,0, 9.8)/3
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = np.clip(self.env.vehicles[id_].ax,0, 9.8)/3
            marker.color = ColorRGBA(0,0,1,1)  
            if id_ in delete_agent_list:
                marker.action = Marker.DELETE
                
            Objects_accel.markers.append(marker)
            
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.id = id_
            marker.type = Marker.CYLINDER
            marker.pose.position.x = self.env.vehicles[id_].x - 1*-np.sin(self.env.vehicles[id_].h)
            marker.pose.position.y = self.env.vehicles[id_].y - 1*np.cos(self.env.vehicles[id_].h)
            marker.pose.position.z = 8 -1/2*np.clip(self.env.vehicles[id_].ax,-9.8, 0)/3
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = -np.clip(self.env.vehicles[id_].ax,-9.8, 0)/3
            marker.color = ColorRGBA(1,0,0,1)  
            if id_ in delete_agent_list:
                marker.action = Marker.DELETE
                
            Objects_brake.markers.append(marker)
            
            text = Marker()
            text.header.frame_id = "base_link"
            text.ns = "text"
            text.id = id_
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.color = ColorRGBA(1, 1, 1, 1)
            text.scale.z = 5
            text.text = str(id_)
            text.pose.position = Point(self.env.vehicles[id_].x, self.env.vehicles[id_].y, 5)
            
            if id_ in delete_agent_list:
                text.action = Marker.DELETE
            
            Texts.markers.append(text)  
                                          
            
        self.sur_pose_plot.publish(Objects)
        self.sur_v_plot.publish(Objects_v)
        self.sur_accel_plot.publish(Objects_accel)
        self.sur_decel_plot.publish(Objects_brake)
        
        self.text_plot.publish(Texts)
        
    def pub_map(self): 
        
        Maps = MarkerArray()
        
        for i in range(len(self.env.map_pt)):
            line_strip = Marker()
            line_strip.type = Marker.LINE_STRIP
            line_strip.id = i
            line_strip.scale.x = 1
            line_strip.scale.y = 0.1
            line_strip.scale.z = 0.1
            
            line_strip.color = ColorRGBA(1.0,1.0,0.0,0.5)
            line_strip.header = Header(frame_id='base_link')

            temp = self.env.map_pt[i]
            for j in range(len(temp)):
                point = Point()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.z = 0
                
                line_strip.points.append(point)
                
            Maps.markers.append(line_strip) 
            
        self.map_plot.publish(Maps)
    
    def set_subscriber(self):
        rospy.Subscriber('/cmd_vel',Twist, self.callback_plot,queue_size=1)   
            
        
    def set_publisher(self):
        
        # Object Cube & ID
        self.sur_pose_plot = rospy.Publisher('/rviz/sur_obj_pose', MarkerArray, queue_size=1)
        self.sur_v_plot = rospy.Publisher('/rviz/sur_v_plot', MarkerArray, queue_size=1)
        self.sur_accel_plot = rospy.Publisher('/rviz/sur_accel_plot', MarkerArray, queue_size=1)
        self.sur_decel_plot = rospy.Publisher('/rviz/sur_decel_plot', MarkerArray, queue_size=1)
        
        self.text_plot = rospy.Publisher('/rviz/text', MarkerArray, queue_size=1)

        # Map Point
        self.map_plot = rospy.Publisher('/rviz/maps', MarkerArray, queue_size=1) 

        
if __name__ == '__main__':

    try:
        f = Simulation()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')


       
    
    
    
        
    
    

