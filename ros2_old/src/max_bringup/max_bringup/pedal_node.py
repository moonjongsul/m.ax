#!/usr/bin/env python3
#
# Copyright 2026 Korea Electronics Technology Institute (KETI)
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
#
# Authors: Jongsul Moon

import sys
import signal
from pynput import keyboard

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class TeleopPedal(Node):
    def __init__(self):
        super().__init__('teleop_pedal_node')
        
        # Launch 파일에서 파라미터 가져오기
        # self.declare_parameter('teleop_topic', '/leader/teleop_start')
        # self.declare_parameter('key', 'b')
        
        # teleop_topic = self.get_parameter('teleop_topic').get_parameter_value().string_value
        # self.toggle_key = self.get_parameter('key').get_parameter_value().string_value.lower()
        
        self.toggle_key = 'b'

        picking_service_name = '/picking_cell/pick'

        self.picking_client = self.create_client(Trigger, picking_service_name)
        self.running = True
        self.keyboard_listener = None

    def call_picking_service(self):
        """토글 키 입력 시 picking 서비스(Trigger) 호출"""
        if not self.picking_client.service_is_ready():
            self.get_logger().warn(
                f"Picking service '{self.picking_client.srv_name}' is not available yet."
            )
            return

        request = Trigger.Request()
        future = self.picking_client.call_async(request)
        future.add_done_callback(self.picking_service_response)

    def picking_service_response(self, future):
        """picking 서비스 응답 처리 콜백"""
        try:
            response = future.result()
            self.get_logger().info(
                f"Picking service responded: success={response.success}, message='{response.message}'"
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Picking service call failed: {e}")

    def on_press(self, key):
        """키보드 키가 눌렸을 때 호출되는 콜백"""
        try:
            # 설정된 토글 키 처리
            if hasattr(key, 'char') and key.char:
                if key.char.lower() == self.toggle_key:
                    self.call_picking_service()
                elif key.char == 'q' or key.char == 'Q':
                    self.get_logger().info("Quitting...")
                    self.running = False
                    return False  # 리스너 종료
        except AttributeError:
            # 특수 키 (예: Ctrl, Shift 등)는 무시
            pass
        return True

    def on_release(self, key):
        """키보드 키가 떼어졌을 때 호출되는 콜백"""
        # 리스너를 계속 유지
        return True

    def start_keyboard_listener(self):
        """전역 키보드 리스너 시작"""
        self.get_logger().info(f"Press '{self.toggle_key}' to toggle teleop start (true/false). Press 'q' to quit.")
        self.get_logger().info("Keyboard listener is active globally (works even when terminal is not focused).")
        
        # 전역 키보드 리스너 시작
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

    def run(self):
        # 전역 키보드 리스너 시작
        self.start_keyboard_listener()

        # 메인 스레드에서는 ROS2 spin
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 키보드 리스너 종료
        if self.keyboard_listener:
            self.keyboard_listener.stop()

def signal_handler(sig, frame, node):
    """시그널 핸들러 (Ctrl+C 처리)"""
    node.running = False
    if node.keyboard_listener:
        node.keyboard_listener.stop()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopPedal()
    
    # 시그널 핸들러 등록 (Ctrl+C 처리)
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, node))
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.running = False
    finally:
        # 키보드 리스너 종료
        if node.keyboard_listener:
            node.keyboard_listener.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()