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