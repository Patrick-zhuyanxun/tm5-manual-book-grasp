import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup.sh"
CHECK = ROOT / "scripts" / "check_env.sh"


def test_scripts_parse():
    for script in (SETUP, CHECK):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_check_is_read_only():
    # Messages may mention these commands; no line may execute one.
    command = re.compile(r"^\s*(sudo|apt-get|rosdep (install|init|update)|rm|uv (sync|venv)|colcon|git submodule update)\b")
    for line in CHECK.read_text().splitlines():
        assert not command.match(line), line


def test_setup_never_upgrades_and_always_asks_before_apt():
    text = SETUP.read_text()
    installs = re.findall(r"sudo apt-get install[^\n]*", text)
    assert installs and all("--no-upgrade" in line for line in installs)
    assert "rosdep install" not in text
    # Every sudo call sits after a confirm() in the same branch.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("sudo "):
            assert any("confirm " in prior for prior in lines[max(0, index - 12):index]), line


def test_unresolvable_tm_moveit_keys_are_skipped():
    text = CHECK.read_text()
    assert 'SKIP_KEYS="gazebo_ros_control warehouse_ros_mongo"' in text
    assert '--skip-keys "$SKIP_KEYS"' in text
