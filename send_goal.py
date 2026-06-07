import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import json
import time

def main():
    rclpy.init()
    node = Node('goal_sender')
    pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
    
    with open("scenarios.json", "r") as f:
        scenarios = json.load(f)
    
    print("목적지 전송 대기 중... (아이작 심의 각 Trial이 시작될 때마다 자동으로 쏩니다)")
    
    for trial_name, data in scenarios.items():
        print(f"[{trial_name}] 로봇 배치를 위해 15초 대기 중...")
        time.sleep(15) # 아이작 심 로딩 및 배치 대기
        
        msg = PoseStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(data['robot_goal'][0])
        msg.pose.position.y = float(data['robot_goal'][1])
        msg.pose.orientation.w = 1.0
        
        pub.publish(msg)
        print(f"[{trial_name}] 목적지 전송 완료: {data['robot_goal']}")
        time.sleep(2) # 짧은 대기

    rclpy.shutdown()

if __name__ == '__main__':
    main()
