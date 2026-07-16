import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import time
import math

class PersistentGoalSender(Node):
    def __init__(self):
        super().__init__('persistent_goal_sender')
        
        # 1. 초기 위치 및 목적지 설정 퍼블리셔/구독자
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.user_goal_callback, 10)
        
        # 2. Nav2 액션 클라이언트
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 3. 상태 및 설정 변수
        self.goal_x, self.goal_y = 2.0, -1.5 
        self.start_x, self.start_y = -6.0, 0.0 
        
        self.is_goal_active = False
        self.user_goal_locked = False
        self.localization_attempts = 0
        self.is_localized = False 

        # 4. 초기화 시퀀스 시작 (타이머를 통해 점진적으로 수행)
        self.get_logger().info('🚀 [Persistent Goal Sender] 로컬라이제이션 대기 모드 (롤백)')
        self.init_timer = self.create_timer(1.0, self.initialization_step)

    def initialization_step(self):
        """초기 위치를 반복적으로 쏘아 RTAB-Map이 위치를 잡게 함"""
        if self.localization_attempts < 5:
            self.get_logger().info(f'📍 [{self.localization_attempts + 1}/5] 초기 위치 전송 중...')
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x = self.start_x
            msg.pose.pose.position.y = self.start_y
            # 목적지를 향하도록 방향 설정 (play_ms.py와 동일하게)
            goal_x, goal_y = 2.0, 0.0 # play_ms.py의 GOAL_X, GOAL_Y
            yaw = math.atan2(goal_y - self.start_y, goal_x - self.start_x)
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            # 공분산(Covariance)을 0으로 설정하여 위치가 완벽함을 알림 (회전 방지)
            msg.pose.covariance = [0.0] * 36
            
            self.init_pub.publish(msg)
            self.localization_attempts += 1
        elif self.localization_attempts < 15:
            # 위치 전송 후 10초간 대기 (Nav2/RTAB-Map 안정화)
            if self.localization_attempts == 5:
                self.get_logger().info('⏳ 로컬라이제이션 안정화 대기 중 (10초)...')
            self.localization_attempts += 1
        else:
            # 모든 준비 완료
            self.get_logger().info('🚗 로컬라이제이션 완료 판단. 주행 타이머로 전환합니다.')
            self.init_timer.cancel()
            self.is_localized = True
            self.timer = self.create_timer(3.0, self.check_and_send_goal)

    def user_goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        if not self.user_goal_locked:
            self.get_logger().info(f'📌 사용자 목적지 고정: ({self.goal_x:.2f}, {self.goal_y:.2f})')
            self.user_goal_locked = True
        self.is_goal_active = False 

    def check_and_send_goal(self):
        if not self.is_goal_active and self.is_localized:
            self.send_goal()

    def send_goal(self):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.goal_x
        goal_msg.pose.pose.position.y = self.goal_y
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f'🎯 목적지 전송: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        self._send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.is_goal_active = True

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn('⚠️ Nav2가 명령을 거절했습니다. 위치 정보가 부족할 수 있습니다.')
                self.is_goal_active = False
                return
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
        except Exception:
            self.is_goal_active = False

    def get_result_callback(self, future):
        self.is_goal_active = False

def main():
    rclpy.init()
    node = PersistentGoalSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
