import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math
import time
import sys

class Nav2Manager(Node):
    def __init__(self):
        super().__init__('nav2_manager')
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def send_initial_pose(self, x, y, gx, gy):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        yaw = math.atan2(gy - y, gx - x)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance = [0.0] * 36
        msg.pose.covariance[0], msg.pose.covariance[7], msg.pose.covariance[35] = 0.05, 0.05, 0.02
        self.init_pub.publish(msg)
        print(f"📍 [ROS2] 초기 위치 전송: ({x}, {y})")

    def send_goal_pose(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        print(f"🎯 [ROS2] 목적지 전송: ({x}, {y})")

def main():
    rclpy.init()
    node = Nav2Manager()
    
    if len(sys.argv) < 2: return

    mode = sys.argv[1]
    if mode == 'init':
        sx, sy, gx, gy = map(float, sys.argv[2:6])
        node.send_initial_pose(sx, sy, gx, gy)
    elif mode == 'goal':
        gx, gy = map(float, sys.argv[2:4])
        node.send_goal_pose(gx, gy)
    
    # 메시지가 확실히 전달되도록 대기 시간을 늘림
    time.sleep(2.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
