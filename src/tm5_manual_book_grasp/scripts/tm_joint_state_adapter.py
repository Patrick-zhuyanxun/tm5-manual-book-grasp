#!/usr/bin/env python3
"""Normalize TM arm states without inventing gripper feedback.

The TM driver publishes the six arm joints.  The project URDF also contains a
real ``motor`` joint for the one-sided gripper.  A missing motor state is
expected while the gripper node is off; it must remain *unknown*, not be
silently substituted with a zero position.  This adapter therefore publishes
only measured arm joints, plus gripper joints that were actually received on
``gripper_joint_states``.  It never commands either device.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint_{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES: Tuple[str, ...] = ("motor", "passive2", "passive_1")


@dataclass(frozen=True)
class JointValue:
    position: float
    velocity: float
    effort: float


def named_joint_values(message: JointState) -> Dict[str, JointValue] | None:
    """Return finite values keyed by name, or ``None`` for a malformed message."""

    if not message.name or len(message.position) != len(message.name):
        return None
    values: Dict[str, JointValue] = {}
    for index, name in enumerate(message.name):
        position = float(message.position[index])
        velocity = float(message.velocity[index]) if index < len(message.velocity) else 0.0
        effort = float(message.effort[index]) if index < len(message.effort) else 0.0
        if not name or not all(math.isfinite(value) for value in (position, velocity, effort)):
            return None
        values[name] = JointValue(position, velocity, effort)
    return values


def normalize_arm_joint_state(message: JointState) -> Dict[str, JointValue] | None:
    """Return a six-axis TM state, accepting named and legacy unnamed messages."""

    values = named_joint_values(message)
    if values is not None:
        if all(name in values for name in ARM_JOINT_NAMES):
            return {name: values[name] for name in ARM_JOINT_NAMES}
        return None

    if message.name or len(message.position) != len(ARM_JOINT_NAMES):
        return None
    arm_values: Dict[str, JointValue] = {}
    for index, name in enumerate(ARM_JOINT_NAMES):
        position = float(message.position[index])
        velocity = float(message.velocity[index]) if index < len(message.velocity) else 0.0
        effort = float(message.effort[index]) if index < len(message.effort) else 0.0
        if not all(math.isfinite(value) for value in (position, velocity, effort)):
            return None
        arm_values[name] = JointValue(position, velocity, effort)
    return arm_values


def merged_joint_state(
    arm_message: JointState,
    gripper_values: Dict[str, JointValue],
) -> JointState | None:
    """Produce an arm state plus only gripper values that were measured."""

    arm_values = normalize_arm_joint_state(arm_message)
    if arm_values is None:
        return None
    output_names = list(ARM_JOINT_NAMES)
    output_names.extend(name for name in GRIPPER_JOINT_NAMES if name in gripper_values)
    values = {**arm_values, **gripper_values}

    output = JointState()
    output.header = arm_message.header
    output.name = output_names
    output.position = [values[name].position for name in output_names]
    output.velocity = [values[name].velocity for name in output_names]
    output.effort = [values[name].effort for name in output_names]
    return output


class TmJointStateAdapter(Node):
    def __init__(self) -> None:
        super().__init__("tm_joint_state_adapter")
        self.declare_parameter("gripper_state_timeout_sec", 0.5)
        self._gripper_state_timeout_sec = float(
            self.get_parameter("gripper_state_timeout_sec").value
        )
        self._gripper_values: Dict[str, JointValue] = {}
        self._gripper_received_monotonic: float | None = None
        self._warned_invalid_arm = False
        self._warned_invalid_gripper = False
        self._publisher = self.create_publisher(JointState, "joint_states", 10)
        self._arm_subscription = self.create_subscription(
            JointState, "input_joint_states", self._arm_callback, 10
        )
        self._gripper_subscription = self.create_subscription(
            JointState, "gripper_joint_states", self._gripper_callback, 10
        )

    def _gripper_callback(self, message: JointState) -> None:
        values = named_joint_values(message)
        if values is None:
            if not self._warned_invalid_gripper:
                self.get_logger().error(
                    "Ignoring malformed gripper joint state; motor state remains unknown."
                )
                self._warned_invalid_gripper = True
            return
        measured = {name: values[name] for name in GRIPPER_JOINT_NAMES if name in values}
        if measured:
            self._gripper_values = measured
            self._gripper_received_monotonic = time.monotonic()

    def _fresh_gripper_values(self) -> Dict[str, JointValue]:
        if self._gripper_received_monotonic is None:
            return {}
        age = time.monotonic() - self._gripper_received_monotonic
        if age > self._gripper_state_timeout_sec:
            return {}
        return self._gripper_values

    def _arm_callback(self, message: JointState) -> None:
        output = merged_joint_state(message, self._fresh_gripper_values())
        if output is None:
            if not self._warned_invalid_arm:
                self.get_logger().error(
                    "Ignoring malformed TM arm state; expected six named joints or "
                    "six finite position values with an empty name array."
                )
                self._warned_invalid_arm = True
            return
        self._publisher.publish(output)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TmJointStateAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
