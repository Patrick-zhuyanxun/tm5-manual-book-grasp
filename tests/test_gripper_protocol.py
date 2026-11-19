import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware" / "gripper_controller" / "gripper_controller.ino"
SCRIPTS = ROOT / "src" / "tm5_manual_book_grasp" / "scripts"


def _firmware_commands():
    return set(re.findall(r"case '([A-Z?])':", FIRMWARE.read_text()))


def test_firmware_handles_every_command_the_coordinator_sends():
    coordinator = (SCRIPTS / "manual_book_grasp.py").read_text()
    sent = set(re.findall(r"data = f?'([A-Z])\b", coordinator))
    assert sent >= {"H", "T", "Y", "O"}, sent
    # G comes from /gripper/gap via gripper_node.py.
    assert "self.send_command(f'G {msg.data}')" in (SCRIPTS / "gripper_node.py").read_text()
    assert sent | {"G"} <= _firmware_commands()


def test_firmware_serial_settings_match_the_node():
    firmware = FIRMWARE.read_text()
    assert "Serial.begin(115200);" in firmware
    assert "declare_parameter('baud_rate', 115200)" in (SCRIPTS / "gripper_node.py").read_text()
    # gripper_node.py terminates each command with '\n'; the sketch splits on it.
    assert "if (c == '\\n')" in firmware


def test_maximum_opening_matches_the_coordinator_limit():
    # gap = 70 * sin(90 deg - openAngle) - GAP_OFFSET must equal the 36.4-mm
    # maximum opening the coordinator assumes.
    import math

    firmware = FIRMWARE.read_text()
    open_angle = float(re.search(r"const int openAngle\s*=\s*(\d+)", firmware).group(1))
    offset = float(re.search(r"const float GAP_OFFSET = ([\d.]+)f;", firmware).group(1))
    gap_max = 70.0 * math.sin(math.radians(90.0 - open_angle)) - offset
    coordinator = (SCRIPTS / "manual_book_grasp.py").read_text()
    assert "declare_parameter('gripper_max_gap_mm', 36.4)" in coordinator
    assert abs(gap_max - 36.4) < 0.05, gap_max
