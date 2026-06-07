#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math
import time

class Nav2EvalHelper(Node):
    def __init__(self):
        super().__init__('nav2_eval_helper')
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.prev_x = None
        self.prev_y = None
        
        self.START_X = -1.0
        self.START_Y = 0.0
        self.GOAL_X = 5.0
        self.GOAL_Y = 0.0
        
        self.first_run = True
        self.get_logger().info("🎯 Nav2 자동화 보조 노드가 준비되었습니다. 리셋 텔레포트를 감시합니다.")

    def odom_callback(self, msg):
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        
        if self.first_run:
            self.send_poses()
            self.first_run = False
            self.prev_x = curr_x
            self.prev_y = curr_y
            return
        
        if self.prev_x is not None and self.prev_y is not None:
            dist = math.dist((curr_x, curr_y), (self.prev_x, self.prev_y))
            # 연속 주행이 아니라 시뮬레이션 순간이동이 일어나면 거리가 확 튑니다.
            if dist > 2.0:
                self.get_logger().info(f"🔄 Isaac Sim 리셋 감지 (이동 거리: {dist:.2f}m). 위치 초기화 및 목적지 재전송 루틴 가동.")
                time.sleep(0.4) # 코스트맵과 맵 매칭이 안정화될 시간 부여
                self.send_poses()
        
        self.prev_x = curr_x
        self.prev_y = curr_y

    def send_poses(self):
        # 1. /initialpose 발행
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.header.frame_id = 'map'
        init_msg.pose.pose.position.x = self.START_X
        init_msg.pose.pose.position.y = self.START_Y
        target_yaw = math.atan2(self.GOAL_Y - self.START_Y, self.GOAL_X - self.START_X)
        init_msg.pose.pose.orientation.z = math.sin(target_yaw / 2.0)
        init_msg.pose.pose.orientation.w = math.cos(target_yaw / 2.0)
        self.init_pose_pub.publish(init_msg)
        
        # 2. /goal_pose 발행
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = self.GOAL_X
        goal_msg.pose.position.y = self.GOAL_Y
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"📤 [동기화 완료] 초기위치: ({self.START_X}, {self.START_Y}) ➡️ 목적지: ({self.GOAL_X}, {self.GOAL_Y})")

def main(args=None):
    rclpy.init(args=args)
    node = Nav2EvalHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    