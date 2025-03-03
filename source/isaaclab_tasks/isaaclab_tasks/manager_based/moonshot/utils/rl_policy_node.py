#!/usr/bin/env python3

from typing import List

import rclpy 
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist
from motion_stack.core.utils.joint_state import JState
from motion_stack.api.ros2.joint_api import JointHandler
from motion_stack.ros2.default_node.lvl1 import DefaultLvl1
from motion_stack.ros2.utils.conversion import ros_to_time

from hero_vehicle_policy import HeroVehiclePolicy

class RlPolicyNode(Node):
    '''
    Subscribes to the topics containing observation data and sends corresponding actions through Motion Stack API.
    '''

    def __init__(self, policy, limbs):
        super().__init__("rl_policy")

        self.cmd_vel_subscriber_  = self.create_subscription(
            Twist, 
            '/cmd_vel', 
            self.cmd_vel_listener_callback, 
            10)
        
        self.base_vel_subscriber_  = self.create_subscription(
            Twist, 
            '/base_vel', 
            self.base_vel_listener_callback, 
            10)

        self.policy = policy
        self.front_wheel = JointHandler(self, limbs[0])
        self.rear_wheel = JointHandler(self, limbs[1])
        self.leg = JointHandler(self, limbs[2])

        self.action = None
        self.base_velocity= None
        self.joint_pos = None
        self.joint_vel = None
        self.command = None
        self.timer = self.create_timer(0.005,self.timer_callback) # 200 Hz

    def cmd_vel_listener_callback(self, msg):
        # vel command only has 3 dims, x, y and rz. 
        self.command = [msg.linear.x, msg.linear.y, msg.angular.z]

    def base_vel_listener_callback(self, msg):
        self.base_velocity = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ]

    def send_action(self, action):
        
        now = self.get_clock().now()

        # check all joints Future if they are ready. 
        if (self.front_wheel.ready.done() and self.rear_wheel.ready.done() and self.leg.ready.done()):

            self.front_wheel.send(
                JState(name=["wheel11_left_joint","wheel11_right_joint"], 
                    time=ros_to_time(now), 
                    velocity = list(action[0:2])
                )
            )
            self.rear_wheel.send(
                JState(name=["wheel12_left_joint","wheel12_right_joint"], 
                    time=ros_to_time(now), 
                    velocity = list(action[2:4])
                )
            )

            # make sure non-actuated joints are at default position, while rest get action commands
            leg_joint_states_dict = {js.name: js.position for js in self.leg.states}
            
            self.leg.send(
                JState(name=["leg1joint1","leg1joint2","leg1joint4","leg1joint6","leg1joint7"], 
                    time=ros_to_time(now), 
                    position = list(action[4:9])
                )
            )
            self.leg.send(
                JState(name=["leg1joint3","leg1joint5"], 
                    time=ros_to_time(now), 
                    position = [leg_joint_states_dict["leg1joint3"], leg_joint_states_dict["leg1joint5"]]
                )
            )
        
        else:
            print("Warning: Not all joints are ready. ")


    def timer_callback(self):

        # concatenate both joint pos and vel into single list
        self.joint_pos = self.front_wheel.states.position + self.rear_wheel.states.position + self.leg.states.position
        self.joint_vel =  self.front_wheel.states.velocity + self.rear_wheel.states.velocity + self.leg.states.velocity
        
        if len(self.joint_pos) != 9: 
            raise ValueError(f'Length of joint position list is not equal to 9: len(joint_pos) = {len(self.joint_pos)}')

        if len(self.joint_vel) != 9:
            raise ValueError(f'Length of joint velocity list is not equal to 9: len(joint_vel) = {len(self.joint_pos)}')

        obs = self.policy.compose_observation(
            base_velocity=self.base_velocity,
            joint_positions=self.joint_pos,
            joint_velocities=self.joint_vel,
            command = self.command,
        )

        action = self.policy.get_action(obs)

        self.send_action(action)

def main(args=None):
    rclpy.init(args=args)

    # pretrained RL policy
    policy = HeroVehiclePolicy(policy_file="policy.pt")
    
    # limb numbers/id in order (front wheel, back wheel, bridge leg)
    limbs = (11, 12, 1)

    node = RlPolicyNode(policy,limbs)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()