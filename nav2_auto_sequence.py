import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math
import time
import sys

class Nav2SequenceNode(Node):
    def __init__(self, sx, sy, gx, gy):
        super().__init__('nav2_sequence_node')
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.sx, self.sy, self.gx, self.gy = sx, sy, gx, gy

    def run_sequence(self):
        # 1. 초기 위치 전송 및 강제 정렬
        print(f"📍 [ROS2] 1. 초기 위치 강제 설정 중... ({self.sx}, {self.sy})")
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.header.frame_id = 'map'
        init_msg.pose.pose.position.x = float(self.sx)
        init_msg.pose.pose.position.y = float(self.sy)
        yaw = math.atan2(self.gy - self.sy, self.gx - self.sx)
        init_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        init_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        init_msg.pose.covariance = [0.0] * 36
        init_msg.pose.covariance[0], init_msg.pose.covariance[7], init_msg.pose.covariance[35] = 0.01, 0.01, 0.01
        
        for _ in range(10): # 1초간 반복 발행하여 확실히 주입
            self.init_pub.publish(init_msg)
            time.sleep(0.1)
        
        # 2. 위치 찾기(Localization) 집중 대기
        # RTAB-Map이 주변 특징점을 잡고 지도를 정렬할 충분한 시간을 줌
        print("⏳ [ROS2] 2. 위치 정밀 매칭 중 (10초 대기)... 제자리에서 지도를 인식합니다.")
        time.sleep(10.0)
        
        # 3. 목적지 전송
        print(f"🎯 [ROS2] 3. 목적지 확정 및 전송: ({self.gx}, {self.gy})")
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(self.gx)
        goal_msg.pose.position.y = float(self.gy)
        goal_msg.pose.orientation.w = 1.0
        
        for _ in range(3): # 목적지 명령 누락 방지
            self.goal_pub.publish(goal_msg)
            time.sleep(0.2)
        
        print("✅ [ROS2] 모든 시퀀스 완료! 로봇이 주행을 시작합니다.")

def main():
    rclpy.init()
    if len(sys.argv) < 5:
        print("Usage: python3 nav2_auto_sequence.py sx sy gx gy")
        return
    sx, sy, gx, gy = map(float, sys.argv[1:5])
    node = Nav2SequenceNode(sx, sy, gx, gy)
    node.run_sequence()
    time.sleep(1.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
