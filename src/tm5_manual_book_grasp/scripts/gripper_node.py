#!/usr/bin/env python3
"""
ROS2 夾爪測試節點
透過 Serial 控制 Arduino 夾爪

用法：
  ros2 run tm5_manual_book_grasp gripper_node.py

指令說明：
  gap <mm>      - 移動到指定 gap（厚度），例如：gap 25
  angle <deg>   - 移動到指定角度，例如：angle 90
  open          - 完全打開
  grip <mm>     - 設定書本厚度並執行兩段式夾取，例如：grip 20
  status        - 查詢當前狀態
  t             - 設定 V_control = HIGH
  y             - 設定 V_control = LOW
  help          - 顯示說明
  quit/exit     - 退出
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import serial
import serial.tools.list_ports
import threading
import time


class GripperController(Node):
    def __init__(self):
        super().__init__('gripper_controller')
        
        # 參數
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 1.0)
        
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.timeout = self.get_parameter('timeout').value
        
        # Serial 連接
        self.ser = None
        self.connect_serial()
        
        # ROS2 訂閱者 - 可以用topic控制
        self.gap_sub = self.create_subscription(
            Float32, '/gripper/gap', self.gap_callback, 10)
        self.angle_sub = self.create_subscription(
            Float32, '/gripper/angle', self.angle_callback, 10)
        self.cmd_sub = self.create_subscription(
            String, '/gripper/command', self.command_callback, 10)
        
        # ROS2 發佈者 - 回報狀態
        self.status_pub = self.create_publisher(String, '/gripper/status', 10)
        
        # 讀取 Serial 回應的線程
        self.running = True
        self.read_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.read_thread.start()
        
        self.get_logger().info('夾爪控制器已啟動')
        self.get_logger().info(f'Serial 連接：{self.serial_port} @ {self.baud_rate}')
    
    def list_available_ports(self):
        """列出可用的 Serial 埠"""
        ports = serial.tools.list_ports.comports()
        self.get_logger().info('可用的 Serial 埠：')
        for port in ports:
            self.get_logger().info(f'  {port.device}: {port.description}')
        return [port.device for port in ports]
    
    def connect_serial(self):
        """連接 Serial"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # Arduino 重置延遲
            self.get_logger().info(f'已連接到 {self.serial_port}')
            return True
        except serial.SerialException as e:
            self.get_logger().error(f'無法連接 {self.serial_port}: {e}')
            self.get_logger().info('正在搜尋可用的 Serial 埠...')
            available = self.list_available_ports()
            
            # 嘗試連接第一個可用的 USB Serial
            for port in available:
                if 'USB' in port or 'ACM' in port:
                    try:
                        self.ser = serial.Serial(
                            port=port,
                            baudrate=self.baud_rate,
                            timeout=self.timeout
                        )
                        time.sleep(2)
                        self.get_logger().info(f'已連接到備用埠 {port}')
                        return True
                    except Exception:
                        continue
            
            self.get_logger().error('無法找到可用的 Serial 埠')
            return False
    
    def send_command(self, cmd: str):
        """發送指令到 Arduino"""
        if self.ser is None or not self.ser.is_open:
            self.get_logger().error('Serial 未連接')
            return False
        
        try:
            full_cmd = cmd.strip() + '\n'
            self.ser.write(full_cmd.encode('utf-8'))
            self.get_logger().info(f'發送指令: {cmd}')
            return True
        except serial.SerialException as e:
            self.get_logger().error(f'發送失敗: {e}')
            return False
    
    def read_serial_loop(self):
        """持續讀取 Serial 回應"""
        while self.running:
            if self.ser is None or not self.ser.is_open:
                time.sleep(0.1)
                continue
            
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.get_logger().info(f'Arduino 回應: {line}')
                        # 發佈狀態
                        msg = String()
                        msg.data = line
                        self.status_pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'讀取錯誤: {e}')
            
            time.sleep(0.01)
    
    # ROS2 Topic 回調
    def gap_callback(self, msg: Float32):
        """處理 gap 指令 (厚度)"""
        self.send_command(f'G {msg.data}')
    
    def angle_callback(self, msg: Float32):
        """處理 angle 指令"""
        self.send_command(f'A {msg.data}')
    
    def command_callback(self, msg: String):
        """處理字串指令"""
        cmd = msg.data.strip().upper()
        if cmd == 'OPEN' or cmd == 'O':
            self.send_command('O')
        elif cmd.startswith('GAP') or cmd.startswith('G '):
            self.send_command(cmd if cmd.startswith('G ') else f'G {cmd[3:].strip()}')
        elif cmd.startswith('ANGLE') or cmd.startswith('A '):
            self.send_command(cmd if cmd.startswith('A ') else f'A {cmd[5:].strip()}')
        elif cmd.startswith('GRIP') or cmd.startswith('H '):
            self.send_command(cmd if cmd.startswith('H ') else f'H {cmd[4:].strip()}')
        elif cmd == 'STATUS' or cmd == 'S':
            self.send_command('S')
        elif cmd == 'T':
            self.send_command('T')
        elif cmd == 'Y':
            self.send_command('Y')
        else:
            self.send_command(msg.data)
    
    def close(self):
        """關閉連接"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


def interactive_mode(node: GripperController):
    """互動式控制模式"""
    print("\n" + "="*50)
    print("ROS2 夾爪控制器 - 互動模式")
    print("="*50)
    print("指令說明：")
    print("  gap <mm>      - 移動到指定 gap（厚度）")
    print("  angle <deg>   - 移動到指定角度")
    print("  open          - 完全打開")
    print("  grip <mm>     - 設定厚度並夾取")
    print("  status        - 查詢狀態")
    print("  t             - 設定 V_control = HIGH")
    print("  y             - 設定 V_control = LOW")
    print("  help          - 顯示說明")
    print("  quit/exit     - 退出")
    print("="*50 + "\n")
    
    while rclpy.ok():
        try:
            user_input = input(">>> ").strip().lower()
            
            if not user_input:
                continue
            
            if user_input in ['quit', 'exit', 'q']:
                print("退出...")
                break
            
            if user_input == 'help' or user_input == 'h':
                print("指令說明：")
                print("  gap <mm>      - 移動到指定 gap（厚度），例如：gap 25")
                print("  angle <deg>   - 移動到指定角度，例如：angle 90")
                print("  open          - 完全打開")
                print("  grip <mm>     - 設定書本厚度並執行兩段式夾取，例如：grip 20")
                print("  status        - 查詢當前狀態")
                print("  t             - 設定 V_control = HIGH")
                print("  y             - 設定 V_control = LOW")
                print("  help          - 顯示說明")
                print("  quit/exit     - 退出")
                continue
            
            if user_input == 'open' or user_input == 'o':
                node.send_command('O')
            elif user_input.startswith('gap '):
                parts = user_input.split()
                if len(parts) >= 2:
                    try:
                        gap = float(parts[1])
                        node.send_command(f'G {gap}')
                    except ValueError:
                        print("錯誤：請輸入有效的數字")
                else:
                    print("用法：gap <mm>")
            elif user_input.startswith('angle '):
                parts = user_input.split()
                if len(parts) >= 2:
                    try:
                        angle = float(parts[1])
                        node.send_command(f'A {angle}')
                    except ValueError:
                        print("錯誤：請輸入有效的數字")
                else:
                    print("用法：angle <deg>")
            elif user_input.startswith('grip '):
                parts = user_input.split()
                if len(parts) >= 2:
                    try:
                        thickness = float(parts[1])
                        node.send_command(f'H {thickness}')
                    except ValueError:
                        print("錯誤：請輸入有效的數字")
                else:
                    print("用法：grip <mm>")
            elif user_input == 'status' or user_input == 's':
                node.send_command('S')
            else:
                # 直接發送原始指令
                node.send_command(user_input.upper())
            
            # 給 Arduino 一些時間回應
            time.sleep(0.2)
            # spin 一下讓回調處理
            rclpy.spin_once(node, timeout_sec=0.1)
            
        except KeyboardInterrupt:
            print("\n退出...")
            break
        except EOFError:
            print("\n退出...")
            break


def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    
    # 使用線程運行 ROS2 spin
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    
    try:
        interactive_mode(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
