#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from std_msgs.msg import String
from rclpy.executors import ExternalShutdownException
from rclpy.exceptions import ROSInterruptException

from motion_stack.ros2.utils.conversion import ros_to_time

import numpy as np

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

class base_vel_sub_pub_node(Node):
    '''
    Converts a base pose time series from MoCap published at topic '/XXX' to
    a base velocity (Twist) published at topic '/base_vel'
    '''

    def __init__(self,base_name):
        super().__init__("base_vel_sub_pub")

        self.base_pose_subscriber_  = self.create_subscription(
            PoseStamped, 
            f'/mocap_{base_name}', 
            self.listener_callback, 
            10)
        
        self.base_vel_publisher_ = self.create_publisher(
            String,
            '/base_vel',
            10
        )
        self.data = None
        self.previous_pose = PoseStamped() 
        self.previous_pose.header.time = ros_to_time(self.get_clock().now())
        self.previous_pose.pose.position.x = 0.0
        self.previous_pose.pose.position.y = 0.0
        self.previous_pose.pose.position.z = 0.0
        self.previous_pose.pose.orientation.x = 0.0
        self.previous_pose.pose.orientation.y = 0.0
        self.previous_pose.pose.orientation.z = 0.0
        self.previous_pose.pose.orientation.w = 1.0  # Identity quaternion
        
        self.timer = self.create_timer(0.005,self.timer_callback) # every 20 ms

    def listener_callback(self, msg):
        self.data = msg

    def publish_base_vel(self):
        now = self.get_clock().now()
        msg = TwistStamped()

        msg.header.time = now
        msg.data.linear = self.base_lin_vel
        msg.data.angular = self.base_ang_vel

        self.base_vel_publisher_.publish(msg)

    def timer_callback(self):

        current_pose = self.data.pose
        dt = current_pose.time - self.previous_pose.time

        self.base_lin_vel = (current_pose.position - self.previous_pose.position)/ dt
        self.base_ang_vel = angular_velocities(self.previous_pose.orientation,current_pose.orientation, dt)

        self.publish_base_vel()

        self.previous_pose = current_pose

def main(args=None):
    rclpy.init(args=args)

    node = base_vel_sub_pub_node(base_name="leg1link4")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException,ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()