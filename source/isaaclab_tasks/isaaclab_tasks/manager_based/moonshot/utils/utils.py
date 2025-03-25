"""Miscellaneous utility functions"""

from typing import List, Dict, Tuple

import re
import os

def find_isaaclab_path(start_path: str = __file__) -> str:
    """
    Finds the full path of the 'IsaacLab' directory by traversing upwards from the start_path.
    
    Args:
        start_path (str): The path to start searching from. Defaults to the current file's location.
    
    Returns:
        str: The full path of the 'IsaacLab' directory.
    
    Raises:
        FileNotFoundError: If 'IsaacLab' directory is not found.
    """
    current_path = os.path.abspath(start_path)
    while current_path != os.path.dirname(current_path):  # Stop when reaching the root directory
        if os.path.basename(current_path) == "IsaacLab" or os.path.basename(current_path) == "SRL-IsaacLab":
            return current_path
        current_path = os.path.dirname(current_path)
    
    raise FileNotFoundError("Could not find 'IsaacLab' directory in the parent hierarchy.")

def sort_wheel_joints(joint_list: List[str]) -> List[str]:
    """
    Sorts wheel joints in ascending numerical order, ensuring left joints appear before right joints.

    Args:
        joint_list (List[str]): List of joint names.

    Returns:
        list of str: Sorted joint names.
    """
    wheel_pattern = re.compile(r"wheel(\d+)_(left|right)_joint")

    # Extract joints with sorting keys
    wheels = []
    for joint in joint_list:
        match = wheel_pattern.match(joint)
        if match:
            wheel_number = int(match.group(1))  # Extract wheel number
            is_right = match.group(2) == "right"  # Right joints come after left
            wheels.append((wheel_number, is_right, joint))

    # Sort by wheel number first, then by left before right
    wheels.sort(key=lambda x: (x[0], x[1]))

    # Return sorted joint names
    return [joint for _, _, joint in wheels]