#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import copy
import math
import bisect
import matplotlib.cm as cm
import matplotlib.animation as animation

from IPython.display import HTML
from utils import *
from agent import agent


import tf
import rospkg
import rospy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray

class Environments(object):
    def __init__(self, course_idx, dt=0.1, min_num_agent=8):
                
        self.spawn_id = 0
        self.vehicles = {}
        self.int_pt_list = {}
        self.min_num_agent = min_num_agent
        self.dt = dt
        self.course_idx = course_idx

        self.initialize()

    def initialize(self, init_num=6):
        self.sensors = {}
        self.pause = False
        filepath = rospy.get_param("file_path")
        Filelist = glob.glob(filepath+"/*info.pickle")
        
        file = Filelist[0]

        with open(file, "rb") as f:
            Data = pickle.load(f)
            
        self.map_pt = Data["Map"]
        self.connectivity = Data["AS"]
        
        for i in range(init_num):
            if i==0:
                CourseList = [[4,1,18], [4,2,25], [4,0,11]]
                self.spawn_agent(target_path = CourseList[self.course_idx], init_v = 0)
            else:
                self.spawn_agent()
    
    def spawn_agent(self, target_path=[], init_v = None):
        
        is_occupied = True
        
        if target_path:
                      
            spawn_cand_lane = target_path[0]
            is_occupied = False
            s_st = 5
            
        else:
            spawn_cand_lane = [10,12,24,17,19]

            s_st = np.random.randint(0,20)
            max_cnt = 10
            while(is_occupied and max_cnt>0):
                
                spawn_lane = np.random.choice(spawn_cand_lane)
                
                is_occupied = False
                for id_ in self.vehicles.keys():
                    if (self.vehicles[id_].lane_st == spawn_lane) and np.abs(self.vehicles[id_].s - s_st) < 25:
                        is_occupied = True    
                        
                max_cnt-=1            
        
        if is_occupied is False:
            if target_path:
                target_path = target_path
                
            else:
                target_path = [spawn_lane]
                spawn_lane_cand = np.where(self.connectivity[spawn_lane]==1)[0]
                
                while(len(spawn_lane_cand)>0):
                    spawn_lane = np.random.choice(spawn_lane_cand)
                    target_path.append(spawn_lane)
                    spawn_lane_cand = np.where(self.connectivity[spawn_lane]==1)[0]
            
            target_pt = np.concatenate([self.map_pt[lane_id][:-1,:] for lane_id in target_path], axis=0)
            self.int_pt_list[self.spawn_id] = {}
            
            for key in self.vehicles.keys():
                intersections = find_intersections(target_pt[:,:3], self.vehicles[key].target_pt[:,:3]) # ((x,y), i, j)
                
                if intersections:
                    self.int_pt_list[self.spawn_id][key] = [(inter, xy[0], xy[1]) for (inter, xy) in intersections]
                    self.int_pt_list[key][self.spawn_id] = [(inter, xy[1], xy[0]) for (inter, xy) in intersections]
                                
            stopline_idx = len(self.map_pt[target_path[0]])-1
            endline_idx = len(self.map_pt[target_path[0]])+len(self.map_pt[target_path[1]])-2
                
            self.vehicles[self.spawn_id] = agent(self.spawn_id, target_path, s_st, target_pt, dt=self.dt, init_v = init_v,
                                                 stoplineidx = stopline_idx, endlineidx = endline_idx)
            self.spawn_id +=1
    
    def delete_agent(self):
        
        delete_agent_list = []
        
        for id_ in self.vehicles.keys():
            if (self.vehicles[id_].target_s[-1]-10) < self.vehicles[id_].s:
                delete_agent_list.append(id_)
            
        return delete_agent_list
                    
    def run(self):
        
        for id_ in self.vehicles.keys():
            if id_ == 0:
                sensor_info = self.vehicles[id_].get_measure(self.vehicles)
                local_lane_info = self.vehicles[id_].get_local_path()

                # 1. control
                ax, steer = self.control(self.vehicles[id_], local_lane_info, sensor_info)

                self.vehicles[id_].step_manual(ax, steer)

            else:
                self.vehicles[id_].step_auto(self.vehicles, self.int_pt_list[id_])


    def respawn(self):
        if len(self.vehicles)<self.min_num_agent:
            self.spawn_agent()

    def control(self, vehicle, lane_info, sensor_info):
        def get_distance(x1, y1, x2, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        steer = vehicle.lateral_controller()
        ax = 0

        vehicle_x = vehicle.x
        vehicle_y = vehicle.y

        # 일정 거리 미만일 때 가속, 아니면 감속
        dist_threshold = 15 # sdv와 일정 거리
        path_dist_threshold = 5 # path와 일정 거리

        # 센서로 측정된 주변 차량과 SDV와의 거리
        dist_arr = [get_distance(x, y, vehicle_x, vehicle_y) for id, x, y, h, vx, vy in sensor_info]
        min_dist = min(dist_arr) if len(dist_arr)>0 else 100

        # 센서로 측정된 주변 차량과 local path와의 거리
        lookahead_dist = vehicle.v * 1
        target_idx = np.where(lane_info[:,0]>=lookahead_dist)[0]
        target_idx = len(lane_info)-1 if len(target_idx)==0 else target_idx[0]

        dist_arr = []
        for id, x, y, h, vx, vy in sensor_info:
            dist_arr = dist_arr + [get_distance(xx,yy,x, y) for xx, yy, hh, rr in lane_info[:target_idx, :]]
        min_dist_path = min(dist_arr) if len(dist_arr)>0 else 100

        if dist_threshold < min_dist and path_dist_threshold < min_dist_path:
            ax = 0.2
        else:
            ax = -0.2

        return ax, steer

if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')


       
    
    
    
        
    
    

