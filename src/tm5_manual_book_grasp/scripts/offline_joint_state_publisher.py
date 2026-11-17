#!/usr/bin/env python3
"""Publish deterministic joint states for the no-robot TM5 visualisation mode.

This node has no subscription and never reads /joint_states.  The launch file
remaps its relative ``joint_states`` output to an isolated offline topic.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class OfflineJointStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("tm5_offline_joint_state_publisher")
        self.declare_parameter(
            "joint_names",
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        )
        self.declare_parameter("positions", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("rate_hz", 10.0)

        self.joint_names = [
            str(name) for name in self.get_parameter("joint_names").value
        ]
        self.positions = [
            float(position) for position in self.get_parameter("positions").value
        ]
        if len(self.joint_names) != len(self.positions):
            raise ValueError("joint_names and positions must have the same length")
        rate_hz = float(self.get_parameter("rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")

        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_state)

    def publish_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.joint_names)
        message.position = list(self.positions)
        self.publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = OfflineJointStatePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ValueError) as exc:
        if isinstance(exc, ValueError):
            print(f"offline joint-state configuration error: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
