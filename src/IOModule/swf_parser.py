import re

def get_unix_start_time(filepath):
    """
    Opens a SWF file and reads the header to find the UnixStartTime.

    Args:
        filepath (str): The full path to the .swf or .cwf trace file.

    Returns:
        int: The Unix Start Time as an integer, or None if not found.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith(';'):
                    # The header is over, no need to read the rest of the file
                    return None
                
                if 'UnixStartTime' in line:
                    # Use regex to find one or more digits after 'UnixStartTime:'
                    match = re.search(r'UnixStartTime:\s*(\d+)', line)
                    if match:
                        return int(match.group(1))
    except FileNotFoundError:
        return None
    
    return None
