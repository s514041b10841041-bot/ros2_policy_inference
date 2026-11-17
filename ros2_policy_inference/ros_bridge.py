"""
Subscribes to the real robot's camera + proprioception topics used as
policy observations, and publishes the policy's predicted EE-offset action.

Each topic is subscribed independently; there is NO time alignment.
get_latest() simply returns the most recent message seen on every topic —
the same approach ros2_data_collector's RosBridge uses, and for the same
reason: robust to topics running at different rates or with
un-synchronised header.stamps.

QoS — camera topics are typically published BEST_EFFORT (sensor-data QoS);
a RELIABLE subscriber silently receives nothing from them, so RGB/depth use
a BEST_EFFORT profile while joint_state/F-T stay RELIABLE.

Gripper state is intentionally NOT subscribed here — see
PolicyInferenceNode, which feeds the policy's own previous gripper output
back in as the next step's gripper observation instead.
"""
import time

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from geometry_msgs.msg import WrenchStamped


class RosBridge:
    def __init__(self, node, cfg, head_rgb_topic=""):
        """head_rgb_topic: optional second RGB camera (CompressedImage), delivered as obs
        "head_rgb"; "" = not subscribed (the default, so policies without it never wait for it)."""
        self._node = node
        self._head_rgb_topic = head_rgb_topic
        self._head_rgb = None

        self._joint = None
        self._rgb = None
        self._depth = None
        self._ft = None

        # topic key -> topic name, for check_health(); last_seen tracked
        # separately (monotonic seconds) so health checks don't depend on
        # any particular clock source.
        self._topics = {
            "joint": cfg.joint_state_topic,
            "rgb": cfg.rgb_topic,
            "depth": cfg.depth_topic,
            "ft": cfg.ft_sensor_topic,
        }
        if head_rgb_topic:
            self._topics["head_rgb"] = head_rgb_topic
        self._last_seen = {}
        # header stamp of the latest message and number of messages received, per topic key
        # (read by obs_sync_check.py)
        self.stamps = {}
        self.counts = {}

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        node.create_subscription(JointState, cfg.joint_state_topic, self._on_joint, 10)
        node.create_subscription(CompressedImage, cfg.rgb_topic, self._on_rgb, sensor_qos)
        node.create_subscription(Image, cfg.depth_topic, self._on_depth, sensor_qos)
        node.create_subscription(WrenchStamped, cfg.ft_sensor_topic, self._on_ft, 10)
        if head_rgb_topic:
            node.create_subscription(CompressedImage, head_rgb_topic, self._on_head_rgb, sensor_qos)

        log = node.get_logger()
        log.info("RosBridge topics:")
        log.info(f"  joint    = {cfg.joint_state_topic}")
        log.info(f"  rgb      = {cfg.rgb_topic}  (CompressedImage, BEST_EFFORT)")
        log.info(f"  depth    = {cfg.depth_topic}  (BEST_EFFORT)")
        log.info(f"  ft       = {cfg.ft_sensor_topic}")
        if head_rgb_topic:
            log.info(f"  head_rgb = {head_rgb_topic}  (CompressedImage, BEST_EFFORT)")

    # ---------- per-topic callbacks ----------

    def _on_joint(self, msg):
        self._joint = {"joint_pos": np.asarray(msg.position, dtype=np.float32)}
        self._last_seen["joint"] = time.monotonic()
        self._count("joint", msg)

    def _on_rgb(self, msg):
        self._rgb = {"rgb": self._compressed_msg_to_array(msg)}
        self._last_seen["rgb"] = time.monotonic()
        self._count("rgb", msg)

    def _on_head_rgb(self, msg):
        self._head_rgb = {"head_rgb": self._compressed_msg_to_array(msg)}
        self._last_seen["head_rgb"] = time.monotonic()
        self._count("head_rgb", msg)

    def _on_depth(self, msg):
        self._depth = {"depth": self._image_msg_to_array(msg)}
        self._last_seen["depth"] = time.monotonic()
        self._count("depth", msg)

    def _on_ft(self, msg):
        f = msg.wrench.force
        t = msg.wrench.torque
        self._ft = {
            "ft_wrench": np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32),
        }
        self._last_seen["ft"] = time.monotonic()

    def _count(self, key, msg):
        self.stamps[key] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def last_seen(self):
        return self._last_seen

    @staticmethod
    def _image_msg_to_array(msg):
        h, w, enc = msg.height, msg.width, msg.encoding

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[..., :3]
            if enc == "bgra8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("16UC1", "mono16"):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()

        if enc == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()

        if enc == "mono8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()

        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1).copy()

    @staticmethod
    def _compressed_msg_to_array(msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)"
            )
        return np.ascontiguousarray(bgr[..., ::-1])

    def get_latest(self):
        """Most recent message from every topic, merged into one dict.
        Returns None until joint + rgb + depth + ft (and head_rgb, when
        subscribed) have each been received at least once."""
        if (self._joint is None or self._rgb is None or self._depth is None
                or self._ft is None or (self._head_rgb_topic and self._head_rgb is None)):
            return None
        out = {}
        out.update(self._joint)
        out.update(self._rgb)
        out.update(self._depth)
        out.update(self._ft)
        if self._head_rgb is not None:
            out.update(self._head_rgb)
        return out

    def received_since(self, t, keys=None):
        """True once every topic in keys has delivered a message after monotonic time t.
        keys=None: joint, rgb and depth, plus head_rgb when it is subscribed."""
        if keys is None:
            keys = ("joint", "rgb", "depth") + (("head_rgb",) if self._head_rgb_topic else ())
        return all(self._last_seen.get(k, -1.0) > t for k in keys)

    def check_health(self, timeout_sec):
        """{topic_name: status} for every subscribed topic, status one of:
          - "ok"           received a message within the last timeout_sec
          - "no_publisher" nothing on the ROS graph is publishing this topic
          - "no_data"      a publisher exists, but no message has arrived yet
                           (e.g. a QoS mismatch — publisher BEST_EFFORT vs a
                           RELIABLE subscription silently drops everything)
          - "stale"        was receiving messages, but not within timeout_sec
                           (e.g. the camera/driver died mid-session)
        """
        now = time.monotonic()
        result = {}
        for key, topic in self._topics.items():
            last = self._last_seen.get(key)
            if last is None:
                publishers = self._node.count_publishers(topic)
                status = "no_data" if publishers > 0 else "no_publisher"
            elif now - last > timeout_sec:
                status = "stale"
            else:
                status = "ok"
            result[topic] = status
        return result
