"""
ROS2 policy-inference node: builds the policy's observation from the robot's
camera + proprioception topics, and runs inference in lockstep with the
consumer (e.g. delta_extension's DeltaActionPolicy) via request_topic/
chunk_topic. See policies/base_policy.py to add a new policy.
"""
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, UInt32, UInt32MultiArray

from .policies import create_policy
from .policies.base_policy import ACTION_DIM
from .preprocessing import resize_image
from .obs_sync_check import ObsSyncCheck
from .ros_bridge import RosBridge

_DEFAULT_PARAMS = dict(
    joint_state_topic="/left_arm/joint_states",
    rgb_topic="/camera_L/color/image_raw/compressed",
    depth_topic="/camera_L/depth/aligned_depth_to_color",
    ft_sensor_topic="/left_ft/wrench_compensated",
    # Which policies/*.py implementation to run — see policies/__init__.py.
    policy_name="dummy",
    topic_timeout_sec=2.0,
    image_size=256,
    initial_gripper_open=True,
    gripper_open_threshold=0.5,
    sync_chunk_size=10,  # actions per chunk; keep equal to the policy's n_action_steps
    request_topic="/il/grasp/request",
    chunk_topic="/il/grasp/action_chunk",
    # Debug: log per chunk whether the observation matches the robot when the chunk starts
    # executing (stale images, RGB/depth from different frames, arm moved) - obs_sync_check.py.
    check_obs_sync=False,
    # Real-Time Chunking: the sim's DeltaActionPolicy publishes [actions executed, last installed
    # chunk seq, actions of it dropped on arrival] (UInt32MultiArray) here every step while its RTC
    # switch is on. While that topic is alive and the policy supports it (diffusion), chunks are
    # full-horizon, stamped with the step they start at, and inpainted onto the previous chunk's
    # unexecuted actions (policies/diffusion/rtc_ddim.py). "" = <request_topic namespace>/exec_step.
    exec_step_topic="",
    # Head RGB camera (CompressedImage), obs "head_rgb". "" = not subscribed. A policy whose
    # config.yaml sets head_rgb_topic (e.g. the le008_* policies) overrides this.
    head_rgb_topic="",
)


@dataclass
class InferenceConfig:
    joint_state_topic: str
    rgb_topic: str
    depth_topic: str
    ft_sensor_topic: str
    policy_name: str
    topic_timeout_sec: float
    image_size: int
    initial_gripper_open: bool
    gripper_open_threshold: float
    sync_chunk_size: int
    request_topic: str
    chunk_topic: str
    check_obs_sync: bool
    exec_step_topic: str
    head_rgb_topic: str


class PolicyInferenceNode(Node):
    def __init__(self, policy=None):
        super().__init__("policy_inference_node")
        cfg = self._load_config()
        self.cfg = cfg

        # The policy is built first: it may declare a head camera topic the bridge must subscribe to.
        self.policy = policy if policy is not None else create_policy(cfg.policy_name)
        self.policy.reset()
        head_rgb_topic = getattr(self.policy, "head_rgb_topic", None) or cfg.head_rgb_topic
        self.bridge = RosBridge(self, cfg, head_rgb_topic=head_rgb_topic)
        self._sync_check = ObsSyncCheck(self, self.bridge, cfg.joint_state_topic) if cfg.check_obs_sync else None

        self._gripper_state = 1.0 if cfg.initial_gripper_open else 0.0

        self._pending = None  # (sequence number, monotonic time the request arrived)
        self._chunk_size = self.policy.sync_chunk_size or cfg.sync_chunk_size
        request_topic = self.policy.request_topic or cfg.request_topic
        chunk_topic = self.policy.chunk_topic or cfg.chunk_topic
        self._chunk_pub = self.create_publisher(Float64MultiArray, chunk_topic, 10)
        self.create_subscription(UInt32, request_topic, self._on_request, 10)

        # Real-Time Chunking state (see exec_step_topic)
        self._exec_step = None          # (actions executed, monotonic arrival)
        self._rtc_prev = None           # (start step, (horizon, ACTION_DIM) normalized) of the last chunk
        self._rtc_delays = deque(maxlen=5)  # actions dropped on arrival (= executed during inference)
        self._rtc_installed_seq = 0
        if getattr(self.policy, "supports_rtc", False):
            exec_step_topic = cfg.exec_step_topic or request_topic.rsplit("/", 1)[0] + "/exec_step"
            self.create_subscription(UInt32MultiArray, exec_step_topic, self._on_exec_step, 50)
            self.get_logger().info(f"Real-Time Chunking available: used while the sim publishes {exec_step_topic}")
        self.create_timer(0.005, self._on_tick)
        self.get_logger().info(
            f"{self._chunk_size}-action chunks on {chunk_topic}, requested on {request_topic}")
        self.create_timer(1.0, self._on_health_check)

    def _load_config(self):
        for name, default in _DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        values = {name: self.get_parameter(name).value for name in _DEFAULT_PARAMS}
        return InferenceConfig(**values)

    def _on_exec_step(self, msg):
        if len(msg.data) < 3:
            return
        step, installed_seq, dropped = (int(v) for v in msg.data[:3])
        self._exec_step = (step, time.monotonic())
        if installed_seq and installed_seq != self._rtc_installed_seq:
            self._rtc_installed_seq = installed_seq
            self._rtc_delays.append(dropped)
            self.get_logger().info(f"chunk {installed_seq}: sim was {dropped} actions ahead on arrival (skipped)")

    def _rtc_active(self):
        return self._exec_step is not None and time.monotonic() - self._exec_step[1] < 1.0

    def _on_request(self, msg):
        self._pending = (int(msg.data), time.monotonic())
        if self._sync_check:
            self._sync_check.on_request()

    def _on_tick(self):
        if self._sync_check:
            self._sync_check.tick()
        if self._pending is None:
            return
        seq, requested_at = self._pending
        # the sim keeps rendering while it waits, so a post-request frame arrives within a few steps
        if not self.bridge.received_since(requested_at):
            if time.monotonic() - requested_at > 1.0:
                self.get_logger().warn(
                    f"chunk {seq}: no observation newer than the request for over 1 s - is the sim publishing?",
                    throttle_duration_sec=5.0,
                )
            return
        latest = self.bridge.get_latest()
        if latest is None:  # a required topic (e.g. F/T, head camera) has never delivered a message
            self.get_logger().warn(
                f"chunk {seq}: waiting for the first message on every observation topic", throttle_duration_sec=5.0)
            return
        obs = self._build_obs(latest)
        if self._sync_check:
            self._sync_check.on_obs(seq, requested_at, obs["joint_pos"])
        if self._rtc_active():
            self._publish_rtc_chunk(seq, obs)
            return
        t0 = time.monotonic()
        self.policy.reset()  # start a fresh chunk from this observation
        actions = np.stack([
            np.asarray(self.policy.predict(obs), dtype=np.float64)
            for _ in range(self._chunk_size)
        ])
        if actions.shape[1] != ACTION_DIM:
            raise ValueError(f"policy returned actions of shape {actions.shape[1:]}, expected ({ACTION_DIM},)")

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label=f"seq={seq}", size=len(actions), stride=actions.size),
            MultiArrayDimension(label="action", size=ACTION_DIM, stride=ACTION_DIM),
        ]
        msg.data = actions.ravel().tolist()
        self._chunk_pub.publish(msg)
        if self._sync_check:
            self._sync_check.on_chunk_sent()
        self._pending = None
        self.get_logger().info(
            f"chunk {seq}: {len(actions)} actions in {(time.monotonic() - t0) * 1e3:.0f} ms, "
            f"first {np.array2string(actions[0], precision=4)}, gripper min {actions[:, 6].min():.3f}"
        )
        # close ends this task; next task starts right after, so feed back "open" either way
        self._gripper_state = 1.0

    def _publish_rtc_chunk(self, seq, obs):
        """Full-horizon chunk starting at the sim's current step, guided by the previous chunk."""
        t0 = time.monotonic()
        start = self._exec_step[0]
        if seq == 1 or self._rtc_prev is None or start < self._rtc_prev[0]:
            self._rtc_prev = None  # new IL episode (the sim's counters restart)
            self._rtc_delays.clear()
        prefix = None
        if self._rtc_prev is not None:
            offset = start - self._rtc_prev[0]  # same horizon row = same sim step
            prefix = self._rtc_prev[1][offset:]
        delay = max(self._rtc_delays) if self._rtc_delays else 0
        frozen = self.policy.first_action_index + delay
        self.policy.reset()
        actions, norm = self.policy.predict_chunk_rtc(obs, prefix, frozen)
        actions = np.asarray(actions, dtype=np.float64)
        self._rtc_prev = (start, norm)

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label=f"seq={seq};start={start}", size=len(actions), stride=actions.size),
            MultiArrayDimension(label="action", size=ACTION_DIM, stride=ACTION_DIM),
        ]
        msg.data = actions.ravel().tolist()
        self._chunk_pub.publish(msg)
        if self._sync_check:
            self._sync_check.on_chunk_sent(rtc=True)
        self._pending = None
        n_prefix = 0 if prefix is None else min(len(prefix), len(norm))
        self.get_logger().info(
            f"chunk {seq} (RTC): {len(actions)} actions from step {start} in {(time.monotonic() - t0) * 1e3:.0f} ms, "
            f"guided by {n_prefix} rows of the previous chunk ({min(frozen, n_prefix)} frozen, est. delay {delay}), "
            f"first {np.array2string(actions[0], precision=4)}, gripper min {actions[:, 6].min():.3f}"
        )
        self._gripper_state = 1.0

    def _build_obs(self, ros_data):
        return {
            "rgb": resize_image(ros_data["rgb"], self.cfg.image_size, is_depth=False),
            "depth": resize_image(ros_data["depth"], self.cfg.image_size, is_depth=True),
            **({"head_rgb": resize_image(ros_data["head_rgb"], self.cfg.image_size, is_depth=False)}
               if "head_rgb" in ros_data else {}),
            "joint_pos": ros_data["joint_pos"],
            "gripper_state": np.float32(self._gripper_state),
            "ft_wrench": ros_data["ft_wrench"],
        }

    def _on_health_check(self):
        """Warn (throttled) about any subscribed topic that has no
        publisher, has never delivered a message, or has gone quiet."""
        messages = {
            "no_publisher": "no publisher found for {topic} — check the topic name / that its node is running",
            "no_data": "publisher exists for {topic} but no message received yet — check QoS (reliability) compatibility",
            "stale": "no message on {topic} in over {timeout:.1f}s — it may have died",
        }
        for topic, status in self.bridge.check_health(self.cfg.topic_timeout_sec).items():
            if status == "ok":
                continue
            self.get_logger().warn(
                messages[status].format(topic=topic, timeout=self.cfg.topic_timeout_sec),
                throttle_duration_sec=5.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
