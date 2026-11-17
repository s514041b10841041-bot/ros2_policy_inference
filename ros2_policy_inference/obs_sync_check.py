"""
Debug check (param check_obs_sync): does each chunk's observation match the robot when the chunk
starts executing? Logs one line per chunk:

  obs-sync chunk 12: OK | image rendered +0 ms vs request step (rgb +0, depth +0) | rgb-depth stamp diff +0 ms |
                     +1 rgb/+1 depth frames after request | arrived after request: rgb 180 joint 20 depth 160 ms |
                     arm moved obs->exec start: 0.12 mm, max joint 0.0003 rad

- "image rendered vs request step": header stamp of each image the policy used minus the stamp of
  the joint state that was current when the request arrived (the sim requests right after the
  previous chunk's last action, so that joint state is the arm at rest). Negative = the image was
  rendered BEFORE the previous chunk finished = STALE. Needs the camera graph and the joint
  publisher on the same clock; if they aren't (stamps seconds apart), it says "clocks differ"
  and only the frame count is left: 0 frames after the request = the image is from before it.
- "rgb-depth stamp diff": the RGB goes through the extension's JPEG re-encode, depth doesn't; a non-
  zero difference means the two images the policy saw were rendered at different times.
- "arm moved obs->exec start": TCP displacement (/left_arm/tcp_pose_world) and largest joint change
  between the joint state the policy used and the first one published after the chunk was sent
  (the sim publishes joint states before applying that step's action, so that is the state the
  first action is applied to). One demo action is ~0.5-1 mm, so well under that = matches.
"""
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

TCP_POSE_TOPIC = "/left_arm/tcp_pose_world"
_STALE_MOVE_MM = 0.5


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class ObsSyncCheck:
    def __init__(self, node, bridge, joint_topic):
        self._node = node
        self._bridge = bridge
        self._log = node.get_logger()
        self._tcp = None            # (arrival monotonic, xyz)
        self._joint = None          # (arrival monotonic, positions)
        self._at_request = None     # image stamps when the request arrived
        self._counts_at_request = {}
        self._joint_stamp_at_request = None
        self._waiting = None        # per-chunk record until the exec-start state arrives
        node.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._on_tcp, 10)
        node.create_subscription(JointState, joint_topic, self._on_joint, 10)
        self._log.info("check_obs_sync on: one 'obs-sync' line per chunk")

    def _on_tcp(self, msg):
        p = msg.pose.position
        self._tcp = (time.monotonic(), np.array([p.x, p.y, p.z]))

    def _on_joint(self, msg):
        self._joint = (time.monotonic(), np.asarray(msg.position, dtype=np.float64))

    def on_request(self):
        self._at_request = dict(self._bridge.stamps)
        self._counts_at_request = dict(self._bridge.counts)
        self._joint_stamp_at_request = self._bridge.stamps.get("joint")

    def on_obs(self, seq, requested_at, obs_joint_pos):
        """Called right after the observation for chunk seq was built."""
        b = self._bridge
        self._pending_obs = dict(
            seq=seq,
            stamps=dict(b.stamps),
            at_request=self._at_request or {},
            joint_stamp_at_request=self._joint_stamp_at_request,
            frames_after={k: b.counts.get(k, 0) - self._counts_at_request.get(k, 0) for k in ("rgb", "depth")},
            arrival_ms={k: (b.last_seen.get(k, requested_at) - requested_at) * 1e3 for k in ("rgb", "depth", "joint")},
            joint=np.asarray(obs_joint_pos, dtype=np.float64),
            tcp=None if self._tcp is None else self._tcp[1].copy(),
        )

    def on_chunk_sent(self, rtc=False):
        """rtc: Real-Time Chunking chunk - the arm keeps moving during inference by design (the sim
        skips the steps it executed meanwhile), so "arm moved" is reported but not flagged."""
        self._waiting = self._pending_obs
        self._waiting["sent_at"] = time.monotonic()
        self._waiting["rtc"] = rtc

    def tick(self):
        w = self._waiting
        if w is None:
            return
        sent = w["sent_at"]
        if self._joint is None or self._joint[0] <= sent or self._tcp is None or self._tcp[0] <= sent:
            if time.monotonic() - sent > 5.0:
                self._log.warn(f"obs-sync chunk {w['seq']}: no joint state/TCP pose after the chunk within 5 s")
                self._waiting = None
            return
        self._waiting = None

        st, req = w["stamps"], w["at_request"]
        fa, ar = w["frames_after"], w["arrival_ms"]
        j0 = w["joint_stamp_at_request"]
        lag = {k: (st[k] - j0) * 1e3 if j0 is not None and k in st else None for k in ("rgb", "depth")}
        same_clock = all(v is not None and abs(v) < 10_000 for v in lag.values())
        if same_clock:
            # 1 ms slack: both stamps come from the same physics step when the image is fresh
            rgb_new, depth_new = lag["rgb"] >= -1.0, lag["depth"] >= -1.0
            render = (f"image rendered {min(lag.values()):+.0f} ms vs request step "
                      f"(rgb {lag['rgb']:+.0f}, depth {lag['depth']:+.0f})")
        else:
            rgb_new, depth_new = fa.get("rgb", 0) > 0, fa.get("depth", 0) > 0
            render = "image vs request step: n/a (camera and joint clocks differ)"
        rgb_depth_ms = (st.get("rgb", 0) - st.get("depth", 0)) * 1e3
        moved_mm = np.linalg.norm(self._tcp[1] - w["tcp"]) * 1e3 if w["tcp"] is not None else float("nan")
        joint_rad = float(np.abs(self._joint[1][: len(w["joint"])] - w["joint"]).max())

        problems = []
        if not rgb_new:
            problems.append("STALE RGB")
        if not depth_new:
            problems.append("STALE DEPTH")
        if abs(rgb_depth_ms) > 1.0:
            problems.append("RGB/DEPTH FROM DIFFERENT FRAMES")
        if moved_mm > _STALE_MOVE_MM and not w["rtc"]:
            problems.append("ARM MOVED")
        line = (
            f"obs-sync chunk {w['seq']}: {', '.join(problems) or 'OK'} | {render} | "
            f"rgb-depth stamp diff {rgb_depth_ms:+.0f} ms | "
            f"+{fa.get('rgb', 0)} rgb/+{fa.get('depth', 0)} depth frames after request | "
            f"arrived after request: rgb {ar['rgb']:.0f} joint {ar['joint']:.0f} depth {ar['depth']:.0f} ms | "
            f"arm moved obs->exec start: {moved_mm:.2f} mm, max joint {joint_rad:.4f} rad"
        )
        (self._log.warn if problems else self._log.info)(line)
