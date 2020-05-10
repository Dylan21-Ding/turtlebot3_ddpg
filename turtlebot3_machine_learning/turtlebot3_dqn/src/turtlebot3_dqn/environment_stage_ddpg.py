#!/usr/bin/env python
# coding=utf-8

#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):   #得到里程计
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation  #方向
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)  #得到yaw航向角

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)#得到弧度值

        heading = goal_angle - yaw #目标和自身角度的差
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2) #得到从当前方向转到面向目标点的转向

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.15
        done = False
        print("scan.ranges:", len(scan.ranges))

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):  #等于正无穷
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):   #判断是否为空值
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2) #障碍物距离最小的地方
        #print("min_distance=", obstacle_min_range)
        obstacle_angle = np.argmin(scan_range)  #给出最小值的下标 找到障碍物在哪
        #print("min_distance_index=", obstacle_angle)
        if min_range > min(scan_range) > 0:
            done = True                       #撞到done置为True

        # 返回所有参数的平方和的平方根
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        if current_distance < 0.2:   #目标与机器人的距离
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done):
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        angle = -pi / 4 + heading + (pi / 8) + pi / 2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        distance_rate = 2 ** (current_distance / self.goal_distance)
        ob_reward = 0

        if obstacle_min_range < 1:
            ob_reward += -5

        reward = round((tr * 5 * distance_rate) + ob_reward, 3)

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        ang_vel = action[0]
        linear = action[1]

        vel_cmd = Twist()  #指线速度角速度的消息类型，通常是用在运动话题/cmd_vel中，被base controller节点监听
        vel_cmd.linear.x = linear  #线速度
        vel_cmd.angular.z = ang_vel #角速度
        print("ang_val = "+str(ang_vel)+" ,linear = "+str(linear))
        self.pub_cmd_vel.publish(vel_cmd)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done)
        print("step_reward=", reward)
        return np.asarray(state), reward, done  #states,reward,isget

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done= self.getState(data)

        return np.asarray(state)