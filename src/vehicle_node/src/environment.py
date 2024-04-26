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

                # 1. convert to global
                global_sensor_info = self.convert_to_global_sensor(sensor_info, self.vehicles[id_])
                global_lane_info = self.convert_to_global_path(local_lane_info, self.vehicles[id_])

                # 2. filtering
                filtered_sensor_info = self.filtering(global_sensor_info)

                # 3. control
                ax, steer = self.control(self.vehicles[id_], global_lane_info, filtered_sensor_info)

                self.vehicles[id_].step_manual(ax, steer)

            else:
                self.vehicles[id_].step_auto(self.vehicles, self.int_pt_list[id_])


    def respawn(self):
        if len(self.vehicles)<self.min_num_agent:
            self.spawn_agent()

    def convert_to_global_path(self, local_path, vehicle):
        rotation_mat = np.array([[np.cos(-vehicle.h), -np.sin(-vehicle.h)],
                                [np.sin(-vehicle.h), np.cos(-vehicle.h)]])

        global_pos = np.matmul(local_path[:,:2], rotation_mat) + np.array([vehicle.x, vehicle.y])
        global_head = np.arctan2(np.sin(local_path[:,2] + vehicle.h), np.cos(local_path[:,2] + vehicle.h))[:,np.newaxis]
        global_R = local_path[:,-1][:,np.newaxis]

        return np.concatenate([global_pos, global_head, global_R], axis=-1)

    def convert_to_global_sensor(self, sensors, vehicle):
        GlobalSensors = []
        rotation_matrix = np.array([[np.cos(vehicle.h), -np.sin(vehicle.h)],
                                    [np.sin(vehicle.h), np.cos(vehicle.h)]])

        for sensor in sensors:
            id_, local_x, local_y, local_h, local_vx, local_vy = sensor

            global_pos = np.dot(rotation_matrix, np.array([local_x, local_y])) + np.array([vehicle.x, vehicle.y])
            global_vel = np.dot(rotation_matrix, np.array([local_vx, local_vy]))

            global_h = local_h + vehicle.h
            global_h = np.arctan2(np.sin(global_h), np.cos(global_h))

            GlobalSensors.append([id_, global_pos[0], global_pos[1], global_h, global_vel[0], global_vel[1]])

        return GlobalSensors

    def filtering(self, sensor_info):
        filtered_sensor_info = []
        # id에 따라 센서 정보를 분류
        for sensor in sensor_info:
            vehicle_id = sensor[0]
            data = np.array(sensor[1:])
            if vehicle_id in self.sensors.keys():
                np.append(self.sensors[vehicle_id], data.reshape(1,5), axis=0)
            else:
                self.sensors[vehicle_id] = [data]

        # noise filtering use by moving average filter
        for key in self.sensors.keys():
            self.sensors[key] = np.array(self.sensors[key])
            filtered_sensor_info.append([key] + list(np.mean(self.sensors[key], axis=0)))
            win_size = min(5, len(self.sensors[key]))
            filtered_sensor_info[-1][0] = moving_average(self.sensors[key][:,0], win_size)
            filtered_sensor_info[-1][1] = moving_average(self.sensors[key][:,1], win_size)
            filtered_sensor_info[-1][2] = moving_average(self.sensors[key][:,2], win_size)
            filtered_sensor_info[-1][3] = moving_average(self.sensors[key][:,3], win_size)
            filtered_sensor_info[-1][4] = moving_average(self.sensors[key][:,4], win_size)

        return filtered_sensor_info

    def control(self, vehicle, lane_info, sensor_info):
        def get_distance(x1, y1, x2, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        steer = vehicle.lateral_controller()
        ax = 0

        # lookahead distance만큼 떨어진 local path 상의 위치
        lookahead_dist = vehicle.v * 1
        dist_arr = np.array([get_distance(x, y, vehicle.x, vehicle.y) for x, y, _, _ in lane_info])
        target_idx = np.where(dist_arr>=lookahead_dist)[0]
        target_idx = len(lane_info)-1 if len(target_idx)==0 else target_idx[0]
        target_idx = -1
        vehicle_pos = [0, 0]
        vehicle_pred_pos = [lane_info[target_idx][0], lane_info[target_idx][1]]

        # 센서로 측정된 주변 차량의 예측 위치 [[x1, y1], [x2, y2]]
        sensor_predictions = [[[x, y],[x+vx, y+vy]] for id, x, y, h, vx, vy in sensor_info]

        #교차 여부를 판단
        pred_intersections = [check_intersection(vehicle_pos, vehicle_pred_pos, [sensor_pos[0][0], sensor_pos[1][0]], [sensor_pred_pos[0][0], sensor_pred_pos[1][0]]) for [sensor_pos, sensor_pred_pos] in sensor_predictions]
        if all(np.array(pred_intersections)==False):
            ax = 0.2
        else:
            print("Intersection")
            ax = -0.2

        return ax, steer

if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')


       
    
    
    
        
    
    

