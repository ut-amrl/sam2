#!/usr/bin/env python3
"""
Extract and subsample images from a ROS2 bag file.

Usage:
    python extract_images.py <bag_path> [options]

Examples:
    python extract_images.py my_bag/                          # Extract all image topics at 1Hz
    python extract_images.py my_bag/ -t /camera/image_raw    # Specific topic
    python extract_images.py my_bag/ -r 5.0                  # 5Hz subsample rate
    python extract_images.py my_bag/ -o /tmp/frames          # Custom output dir
    python extract_images.py my_bag/ --list-topics            # List available topics
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    print("ERROR: ROS2 Python packages not found.")
    print("Make sure you have sourced your ROS2 workspace:")
    print("  source /opt/ros/<distro>/setup.bash")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not found. Install with:")
    print("  pip install opencv-python")
    sys.exit(1)


# ── Supported image message types ─────────────────────────────────────────────

IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


def _detect_storage_id(bag_path: str) -> str:
    """Detect whether a bag uses mcap or sqlite3 storage."""
    bag_dir = Path(bag_path)
    if any(bag_dir.glob("*.mcap")):
        return "mcap"
    return "sqlite3"


def get_reader(bag_path: str):
    """Open a ROS2 bag for reading."""
    storage_id = _detect_storage_id(bag_path)
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def list_topics(bag_path: str):
    """Print all topics in the bag and their message types."""
    reader = get_reader(bag_path)
    topic_types = reader.get_all_topics_and_types()
    print(f"\nTopics in '{bag_path}':\n")
    print(f"  {'Topic':<50} {'Type'}")
    print(f"  {'-'*50} {'-'*40}")
    for t in topic_types:
        marker = "  [image]" if t.type in IMAGE_TYPES else ""
        print(f"  {t.name:<50} {t.type}{marker}")
    print()


def image_msg_to_cv2(msg, msg_type: str):
    """Convert a ROS image message to a cv2 BGR image."""
    if msg_type == "sensor_msgs/msg/CompressedImage":
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    # sensor_msgs/msg/Image
    encoding = msg.encoding.lower()
    dtype_map = {
        "8uc1": np.uint8,  "8uc3": np.uint8,  "8uc4": np.uint8,
        "16uc1": np.uint16, "32fc1": np.float32,
        "rgb8": np.uint8,  "rgba8": np.uint8,
        "bgr8": np.uint8,  "bgra8": np.uint8,
        "mono8": np.uint8, "mono16": np.uint16,
    }
    channels_map = {
        "8uc1": 1, "8uc3": 3, "8uc4": 4,
        "16uc1": 1, "32fc1": 1,
        "rgb8": 3, "rgba8": 4,
        "bgr8": 3, "bgra8": 4,
        "mono8": 1, "mono16": 1,
    }

    dtype = dtype_map.get(encoding, np.uint8)
    channels = channels_map.get(encoding, 3)

    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    if channels == 1:
        img = arr.reshape((msg.height, msg.width))
    else:
        img = arr.reshape((msg.height, msg.width, channels))

    # Convert to BGR for cv2.imwrite
    if encoding in ("rgb8",):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ("rgba8",):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding in ("bgra8", "8uc4"):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding in ("mono8", "8uc1"):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif encoding in ("mono16", "16uc1"):
        img = (img / 256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif encoding in ("32fc1",):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def extract_images(
    bag_path: str,
    topics: list[str] | None,
    output_dir: str,
    rate_hz: float,
    image_format: str,
    quality: int,
    start_sec: float | None,
    end_sec: float | None,
):
    """Main extraction loop."""
    reader = get_reader(bag_path)
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    # Auto-detect image topics if none specified
    if not topics:
        topics = [t.name for t in topic_types if t.type in IMAGE_TYPES]
        if not topics:
            print("ERROR: No image topics found in bag. Use --list-topics to inspect.")
            sys.exit(1)
        print(f"Auto-detected image topics: {topics}")

    # Validate requested topics
    for topic in topics:
        if topic not in type_map:
            print(f"ERROR: Topic '{topic}' not found in bag.")
            sys.exit(1)
        if type_map[topic] not in IMAGE_TYPES:
            print(f"WARNING: Topic '{topic}' is '{type_map[topic]}' — may not be an image.")

    # Set up storage filter
    storage_filter = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(storage_filter)

    # Create per-topic output dirs and state tracking
    os.makedirs(output_dir, exist_ok=True)
    topic_dirs = {}
    last_saved_ns = {}   # topic -> last saved timestamp (nanoseconds)
    counters = {}        # topic -> frame count

    bag_name = Path(bag_path).stem
    for topic in topics:
        tdir = os.path.join(output_dir, bag_name)
        os.makedirs(tdir, exist_ok=True)
        topic_dirs[topic] = tdir
        last_saved_ns[topic] = -1
        counters[topic] = 0

    interval_ns = int(1e9 / rate_hz)  # subsample interval in nanoseconds
    start_ns = int(start_sec * 1e9) if start_sec is not None else None
    end_ns   = int(end_sec   * 1e9) if end_sec   is not None else None

    # cv2 write params
    write_params = []
    if image_format == "jpg":
        write_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif image_format == "png":
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, quality // 11))]

    print(f"\nExtracting to '{output_dir}' at {rate_hz} Hz ...\n")

    total_saved = 0

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()

        if start_ns is not None and timestamp_ns < start_ns:
            continue
        if end_ns is not None and timestamp_ns > end_ns:
            continue

        # Subsample check
        if last_saved_ns[topic_name] >= 0:
            if timestamp_ns - last_saved_ns[topic_name] < interval_ns:
                continue

        # Deserialize
        msg_type_str = type_map[topic_name]
        try:
            msg_class = get_message(msg_type_str)
            msg = deserialize_message(data, msg_class)
        except Exception as e:
            print(f"  WARNING: Failed to deserialize message on '{topic_name}': {e}")
            continue

        # Convert to cv2
        try:
            img = image_msg_to_cv2(msg, msg_type_str)
        except Exception as e:
            print(f"  WARNING: Failed to convert image on '{topic_name}': {e}")
            continue

        if img is None:
            print(f"  WARNING: Empty image on '{topic_name}' at t={timestamp_ns}")
            continue

        # Save
        frame_idx = counters[topic_name]
        filename = f"frame_{frame_idx:06d}_{timestamp_ns}.{image_format}"
        filepath = os.path.join(topic_dirs[topic_name], filename)

        cv2.imwrite(filepath, img, write_params)

        counters[topic_name] += 1
        last_saved_ns[topic_name] = timestamp_ns
        total_saved += 1

        if frame_idx % 50 == 0:
            print(f"  [{topic_name}] frame {frame_idx:06d}  t={timestamp_ns/1e9:.3f}s")

    print(f"\nDone. Saved {total_saved} frames total.")
    for topic in topics:
        print(f"  {topic}: {counters[topic]} frames → {topic_dirs[topic]}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and subsample images from a ROS2 bag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument(
        "-t", "--topics", nargs="+", metavar="TOPIC",
        help="Image topic(s) to extract (default: auto-detect all image topics)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="extracted_images",
        help="Output directory (default: ./extracted_images)",
    )
    parser.add_argument(
        "-r", "--rate", type=float, default=1.0, metavar="HZ",
        help="Target extraction rate in Hz (default: 1.0)",
    )
    parser.add_argument(
        "-f", "--format", choices=["jpg", "png"], default="jpg",
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=95, metavar="0-100",
        help="JPEG quality or PNG compression proxy 0-100 (default: 95)",
    )
    parser.add_argument(
        "--start", type=float, default=None, metavar="SEC",
        help="Start time offset in seconds from bag start",
    )
    parser.add_argument(
        "--end", type=float, default=None, metavar="SEC",
        help="End time offset in seconds from bag start",
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="List all topics in the bag and exit",
    )

    args = parser.parse_args()

    bag_path = str(Path(args.bag_path).resolve())
    if not os.path.exists(bag_path):
        print(f"ERROR: Bag path not found: {bag_path}")
        sys.exit(1)

    if args.list_topics:
        list_topics(bag_path)
        sys.exit(0)

    extract_images(
        bag_path=bag_path,
        topics=args.topics,
        output_dir=args.output_dir,
        rate_hz=args.rate,
        image_format=args.format,
        quality=args.quality,
        start_sec=args.start,
        end_sec=args.end,
    )


if __name__ == "__main__":
    main()