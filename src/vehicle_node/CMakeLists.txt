cmake_minimum_required(VERSION 3.0.2)
project(vehicle_node)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
  visualization_msgs
)

catkin_package(

 CATKIN_DEPENDS rospy std_msgs

)

