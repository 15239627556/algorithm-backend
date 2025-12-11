import math
import os
import sys
import json
import time
from deepmerge import always_merger
import numpy as np


def is_jsonable(x):
    """Check if x can be serialized by json"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def serialize_non_json_fields(data):
    """Recursively convert non-serializable fields to repr strings"""
    if isinstance(data, dict):
        return {k: serialize_non_json_fields(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_non_json_fields(v) for v in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    elif isinstance(data, np.int64):
        return int(data)
    elif not is_jsonable(data):
        return repr(data)  # Or use a custom tag like "__non_serializable__" + repr(data)
    else:
        return data


def deserialize_non_json_fields(data):
    """Convert back any repr string to Python object"""
    if isinstance(data, dict):
        return {k: deserialize_non_json_fields(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_non_json_fields(v) for v in data]
    elif isinstance(data, str):
        try:
            # Try converting from repr() string
            return eval(data)
        except:
            return data
    else:
        return data


if sys.platform == 'win32':
    import msvcrt

    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def lock_file(f):
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock_file(f):
        fcntl.flock(f, fcntl.LOCK_UN)


def safe_json_write(file_path, update_data, max_retries=3):
    """Thread-safe JSON file modification function with support for non-serializable data."""
    for _ in range(max_retries):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)

            with open(file_path, 'r+', encoding='utf-8') as f:
                lock_file(f)

                data = {}
                if os.path.getsize(file_path) > 0:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in file: {file_path}. Initializing as empty dictionary.")
                        data = {}

                # Preprocess update_data to make it serializable
                processed_update = serialize_non_json_fields(update_data)
                always_merger.merge(data, processed_update)

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)

                unlock_file(f)
                return True
        except (BlockingIOError, json.JSONDecodeError) as e:
            print(e)
            time.sleep(0.1)
    return False


def safe_json_read(file_path):
    """Read and deserialize the JSON file, restoring non-serializable values."""
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return deserialize_non_json_fields(data)
        except json.JSONDecodeError:
            print(f"Failed to load JSON from {file_path}")
            return {}