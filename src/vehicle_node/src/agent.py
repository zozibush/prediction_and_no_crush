import numpy as np
import bisect
from utils import *


class agent():
    def __init__(self, spawn_id, lane, s_st, target_pt, dt=0.1, init_v = None, target_v = None, stoplineidx = None, endlineidx = None):
        
        self.id = spawn_id
        self.lane_st = lane[0]
        self.dt = dt
        
        self.target_pt = target_pt[:,:3]
        self.target_s = np.cumsum(np.linalg.norm(self.target_pt[1:,:2]-self.target_pt[:-1,:2], axis=-1))
        self.target_s = np.insert(self.target_s, 0, 0, axis=0)
        self.target_radius = target_pt[:,-1]
        
        self.stoplineidx = stoplineidx
        self.endlineidx = endlineidx
        
        self.s = s_st
        self.d = 0
        self.s_idx = bisect.bisect_left(self.target_s, self.s)
        
        self.x, self.y = get_cartesian(self.s, self.d, self.target_pt[:,0],self.target_pt[:,1], self.target_s)
        self.h = self.get_heading()
        
        if init_v is not None:
            self.v = init_v
        else:
            self.v = np.random.randint(8,13)        
        
        self.target_v = np.random.randint(self.v,15) 
        
        self.ax = 0
        self.steer = 0
        
        self.front_margin = 5
        self.rear_margin = 5
        
    def get_heading(self):
        ratio = (self.s-self.target_s[self.s_idx])/(self.target_s[self.s_idx+1]-self.target_s[self.s_idx])
        h = self.target_pt[self.s_idx,2] + ratio*(self.target_pt[self.s_idx+1,2]-self.target_pt[self.s_idx,2])
        h = np.arctan2(np.sin(h), np.cos(h))
        
        return h
        
    def get_local_path(self):
        
        target_local_path = self.target_pt[self.s_idx:self.s_idx+50,:] - np.array([self.x, self.y, self.h])
        rotation_mat = np.array([[np.cos(self.h), -np.sin(self.h)], 
                                 [np.sin(self.h), np.cos(self.h)]])
        
        target_local_pos = np.matmul(target_local_path[:,:2], rotation_mat)
        target_local_head = np.arctan2(np.sin(target_local_path[:,2]), np.cos(target_local_path[:,2]))[:,np.newaxis]
        target_local_R = self.target_radius[self.s_idx:self.s_idx+50][:,np.newaxis]
        
        return np.concatenate([target_local_pos, target_local_head, target_local_R], axis=-1)
                
        
    def get_measure(self, vehicles):
        
        Sensors = []
        
        for id_ in vehicles.keys():
            if id_ == self.id:
                continue
                
            noise_x = np.random.randn()*0.5
            noise_y = np.random.randn()*0.3
            noise_h = np.random.randn()*0.05
            noise_vx = np.random.randn()*0.1
            noise_vy = np.random.randn()*0.2
            
            
            veh_x = vehicles[id_].x - self.x
            veh_y = vehicles[id_].y - self.y
            veh_h = vehicles[id_].h - self.h
            veh_vx = vehicles[id_].v * np.cos(veh_h) - self.v
            veh_vy = vehicles[id_].v * np.sin(veh_h)
            
            
            veh_pos_local_x = veh_x*np.cos(self.h) + veh_y*np.sin(self.h) + noise_x
            veh_pos_local_y = -veh_x*np.sin(self.h) + veh_y*np.cos(self.h) + noise_y
            veh_pos_local_h = veh_h + noise_h
            veh_pos_local_h = np.arctan2(np.sin(veh_pos_local_h), np.cos(veh_pos_local_h))
            veh_vel_local_x = veh_vx + noise_vx
            veh_vel_local_y = veh_vy + noise_vy
            
            
            if np.abs(np.arctan2(veh_pos_local_y, veh_pos_local_x)) <=60/57.3:
                Sensors.append([id_, veh_pos_local_x, veh_pos_local_y, veh_pos_local_h, veh_vel_local_x, veh_vel_local_y])            
            
        return Sensors
    
    def lateral_controller(self):
        
        path = self.get_local_path()
        self.local_path = path
        
        if len(path)==0:
            delta = 0
            
        else:
            lookahead_dist = self.v*1

            target_idx = np.where(path[:,0]>=lookahead_dist)[0]

            if len(target_idx)==0:
                target_idx = len(path)-1
            else:
                target_idx = target_idx[0]

            target_x = path[target_idx, 0]
            target_y = path[target_idx, 1]
                    
            r = (target_x**2+target_y**2)/(2*target_y+1e-2)

            delta = np.arctan2(6/r, 1)
                
        return delta 
    
    def longitudianal_controller(self, vehicles, int_pt_list):
        
        ax_list = self.get_a_agent(vehicles, int_pt_list)                    
        
        # add non-front vehicle case
        ax_tarv = self.IDM((1e3, 0, -1e2))               
        ax_list.append(ax_tarv)
        
        # add curvature case
        ax_curv = self.get_a_curvature() 
        ax_list.append(ax_curv)
        
        min_idx = np.argmin(np.array(ax_list))
        ax = ax_list[min_idx]
            
        return np.clip(ax, -5,5)
    
    def step_manual(self, ax, steer):
        
        steer = self.lateral_controller()
            
        self.v += ax*self.dt
        self.v = np.clip(self.v, 0, 30)
        self.h += steer*self.dt
        
        self.x += np.cos(self.h)*self.v*self.dt
        self.y += np.sin(self.h)*self.v*self.dt
                
        self.s, self.d = get_frenet(self.x, self.y, self.target_pt[:,0], self.target_pt[:,1], self.target_s)        
        self.s_idx = bisect.bisect_left(self.target_s, self.s)   
        
    def step_auto(self, vehicles, int_pt_list):
        
        delta = self.lateral_controller()
        self.steer = delta
        
        ax = self.longitudianal_controller(vehicles, int_pt_list)
        dax = np.clip(ax-self.ax, -1,1)
        self.ax +=dax
        self.ax = np.clip(self.ax, -self.v / self.dt, 5)
              
        self.v += self.ax*self.dt

        if self.v < 1:
            delta = np.clip(delta, -10/57.3, 10/57.3)
            
        self.h += delta*self.dt
        self.x += np.cos(self.h)*self.v*self.dt
        self.y += np.sin(self.h)*self.v*self.dt
                
        self.s, self.d = get_frenet(self.x, self.y, self.target_pt[:,0], self.target_pt[:,1], self.target_s)        
        self.s_idx = bisect.bisect_left(self.target_s, self.s)        
            
    def IDM(self, front_info):
        # Compute the desired dynamic distance
        T = 1.5 # safe time headway(s)
        s0 = 5 # minimum desired net distance(m)
        delta = 4 #acceleration exponenet
        a = 0.73 # Maximum vehicle acceleration (m/s^2) # p에 따라 바꿔야 함.
        b = 1.67 # Comfortable deceleration (m/s^2)
        
        ego2int, tar2int, delta_v =  front_info
        s = ego2int-tar2int + 1e-5
                
        v = self.v
        v0 = self.target_v + 1e-5
        
        s_star = s0 + max(0, v * T + (v * delta_v) / (2 * np.sqrt(a * b)))

        # Compute the acceleration
        acceleration = a * (1 - (v / v0)**delta - (s_star / s)**2)
        
        return np.clip(acceleration, -5, 5)
    
    def get_a_curvature(self):
        
        max_ay = 1.2
        lookahead_s = np.clip(self.v*2, 15, 70) + self.s
        end_idx = bisect.bisect_left(self.target_s, lookahead_s)
        
        if self.s_idx == end_idx:
            min_r = 1e3
            min_r_idx = 0
            ax = 0
            
        else:
            min_r_idx = np.argmin(self.target_radius[self.s_idx:end_idx])
            min_r = self.target_radius[self.s_idx+min_r_idx] 
            v_max = np.sqrt(max_ay*np.abs(min_r))
            ax = (v_max-self.v)/self.dt
            ax = np.clip((v_max**2 - self.v**2)/(2*(self.target_s[self.s_idx+min_r_idx]- self.s)), -10, 10)
        
        return ax
    
    def get_a_agent(self,vehicles, int_pt_list):
        
        ax_list = []  
        
        # get interest agents info
        front_id_list = self.get_front_id(vehicles, int_pt_list)
            
        for front_info in front_id_list:
            
            if front_info[-1] == 0: 
                # 전방 ego lane 일부분을 차지
                ax = self.IDM(front_info[1:4])
                
            elif front_info[-1] == 3:
                ## merge
                ax = self.IDM(front_info[1:4])   
                
            else:
                # cross
                tarid = front_info[0]
                tar2int = np.clip(front_info[2],0,1e2) + self.rear_margin ## 후방이 통과까지 걸리는 거리
                tarv_expect = np.sqrt((np.clip(2*vehicles[tarid].ax*tar2int,0,100) + vehicles[tarid].v**2))
                tarTTI = np.max([(tarv_expect-vehicles[tarid].v) / (vehicles[tarid].ax + 1e-5), tar2int/(vehicles[tarid].v+1e-2)])
                ego2int = front_info[1] - self.front_margin*2 # 전방이 진입까지 걸리는 거리
                ax = np.clip((2*ego2int - 2*self.v*tarTTI)/(tarTTI**2+1e-4), -10, 3) 

            ax_list.append(ax)    
        
        return ax_list   
    
    def get_front_id(self, Vehicles, int_pt_list):
        
        """
        충돌 가능성이 있는 agent's id를 탐색
        """
        ROI_range = 50
        Front_id_list = []
        
        for id_ in Vehicles.keys():
            
            tar = Vehicles[id_]
            
            front_flag = -1  # front_flag : -1 = None / 0 = front / 1, 2 = cross / 3 = merge  
            
            if self.id == tar.id:
                continue
            
            ## Check front agents
            s, d = get_frenet(tar.x, tar.y, self.target_pt[:,0],self.target_pt[:,1], self.target_s)
            ego2int = s - self.s            
            
            if d < 2.4 and ego2int >= 0:
                tar2int = 0
                dh = (self.h - tar.h)
                rel_v = self.v - tar.v*np.cos(dh)
                front_flag = 0
                Front_id_list.append( [ id_, ego2int-self.front_margin, tar2int, rel_v, self.v, tar.v, front_flag ] )
                continue
            
            ## Check intersection agents
            # Get first intersection point between two agents
            inter_idx = None
            if id_ in int_pt_list.keys():
                for int_pt in int_pt_list[id_]:
                    if int_pt[1] >= (self.s_idx) and int_pt[2] >= (tar.s_idx): 
                        inter_idx = [int_pt[1], int_pt[2]]
                        break
            
            if inter_idx:
                ego2int = self.target_s[inter_idx[0]] - self.s
                tar2int = tar.target_s[inter_idx[1]] - tar.s

                egoTTI = ego2int / np.clip( self.v , 3, 30)
                tarTTI = tar2int / np.clip(tar.v , 3, 30)       
                
                if tar2int > ROI_range or tarTTI > 20:
                    continue
                
                else:
                    ego_angle_int = self.target_pt[inter_idx[0],2]
                    tar_angle_int = tar.target_pt[inter_idx[1],2]
                    inter_angle = ( ego_angle_int - tar_angle_int) 
                    inter_angle = np.arctan2(np.sin(inter_angle), np.cos(inter_angle))  
                    
                    isinsidetar = tar.target_s[tar.stoplineidx] < tar.s and \
                            tar.s <= tar.target_s[tar.endlineidx]  # True if target agent is inside intersection area.

                    rel_v = self.v - tar.v
                    
                    # 같은 교점으로 접근. merging, cross 
                    if np.abs(inter_angle) * 57.3 > 30:
                        ## cross 
                        if (tarTTI < egoTTI): ## target이 우선순위.
                            front_flag = 1
                            Front_id_list.append( [ id_, ego2int, tar2int, rel_v, self.v, tar.v, front_flag ] )
                        
                        if (isinsidetar and (tar2int < ego2int)):
                            front_flag = 2
                            Front_id_list.append( [ id_, ego2int, tar2int, rel_v, self.v, tar.v, front_flag ] )
                    
                    else:
                        if ego2int > 2.5 and tar2int > 2.5 and (tar2int < ego2int): 
                            ## merge
                            front_flag = 3
                            Front_id_list.append( [ id_, ego2int, tar2int, rel_v, self.v, tar.v, front_flag ] )
                            
        return Front_id_list
    