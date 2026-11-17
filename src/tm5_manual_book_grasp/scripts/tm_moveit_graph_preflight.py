#!/usr/bin/env python3
"""Reject startup when another MoveIt or TM driver is already in this domain."""

from __future__ import annotations

import time
from typing import Iterable, Sequence, Tuple

import rclpy
from rclpy.node import Node


NodeIdentity = Tuple[str, str]

# These nodes own global actions/topics used by this launch.  Running a second
# copy makes RViz alternate between independent planning scenes and allows two
# servers to answer the same MoveIt action request.
CONFLICTING_NODE_NAMES = {
    "move_group",
    "tm_driver",
    "tm_driver_node",
    "tm5_offline_joint_state_publisher",
    "robot_state_publisher",
    "static_transform_publisher",
}


def fully_qualified_name(name: str, namespace: str) -> str:
    namespace = namespace.rstrip("/")
    return f"{namespace}/{name}" if namespace else f"/{name}"


def find_conflicts(nodes: Sequence[NodeIdentity]) -> list[str]:
    conflicts = []
    for name, namespace in nodes:
        if name in CONFLICTING_NODE_NAMES or name.startswith("move_group_private_"):
            conflicts.append(fully_qualified_name(name, namespace))
    return sorted(set(conflicts))


class MoveItGraphPreflight(Node):
    def __init__(self) -> None:
        super().__init__("tm_moveit_graph_preflight")
        self.declare_parameter("discovery_timeout_sec", 1.5)

    def check(self) -> list[str]:
        timeout = float(self.get_parameter("discovery_timeout_sec").value)
        deadline = time.monotonic() + timeout
        # DDS discovery is asynchronous.  Accumulate observations for the full
        # interval so a late-discovered stale process cannot be missed.
        observed: set[NodeIdentity] = set()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            observed.update(self.get_node_names_and_namespaces())
        observed.discard((self.get_name(), self.get_namespace()))
        return find_conflicts(tuple(observed))


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveItGraphPreflight()
    exit_code = 0
    try:
        conflicts = node.check()
        if conflicts:
            node.get_logger().fatal(
                "Refusing startup: another MoveIt/TM runtime is already present: "
                + ", ".join(conflicts)
            )
            exit_code = 2
        else:
            node.get_logger().info(
                "ROS graph preflight passed: no existing MoveIt/TM runtime found."
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
