#!/usr/bin/env python3
"""
Replay one episode's recorded actions from a LeRobot dataset (e.g. Le-007-turbo) into Isaac Sim and
compare the sim's gripper trajectory with the one recorded in the episode.

Checks whether the sim reproduces the demo when fed its actions open-loop, i.e. what a policy that
predicts those actions perfectly would get. Adapted from delta_01's tools/action_replay/replay_actions.py
(real-robot .npz samples); this one reads the LeRobot parquet files directly.

Action convention (verified on Le-007-turbo with URDF FK of observation.state): action[t] is the
gripper_center pose change from frame t to t+1 in the sim world frame - position offset, plus rotation
vector applied by left multiplication. The Delta Extension's DeltaActionPolicy applies the same offsets,
but reads the rotation as XYZ Euler angles, so each rotation vector is converted before it is sent.

Protocol (delta_extension tasks/policies/delta_action.py): when its action queue runs dry the sim
publishes a request number on /il/grasp/request (std_msgs/UInt32); this script answers on
/il/grasp/action_chunk with the next --chunk-size actions (std_msgs/Float64MultiArray, N x 7 flattened,
layout.dim[0].label = "seq=<request number>", as ros2_policy_inference's node does). The sim executes
one action per physics step, so the demo's timing (one action per 1/60 s) holds in sim time however
slowly the sim runs.

  1. In Isaac Sim: Arm Sync Mode No Sync, Fan Grasp Mode IL, Publish Policy Observations on, press Play.
     Do NOT run a policy node: two publishers on /il/grasp/action_chunk would mix.
  2. Start this script; it waits for the sim's first chunk request.
  3. Run the task in the Delta Extension panel. The script answers every request with the next chunk,
     records /left_arm/tcp_pose_world, and writes <out>.csv and <out>.html.

Both trajectories are compared as displacement from their own start, in the world frame, so the sim's
start pose does not need to match the episode's. The fan is wherever the sim scene puts it, not where it
was in the demo, so a successful grasp is not expected - this checks the motion only.
"""
import argparse
import glob
import html
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.dataset as pads
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, UInt32

REQUEST_TOPIC = "/il/grasp/request"           # IL_GRASP_REQUEST_TOPIC
CHUNK_TOPIC = "/il/grasp/action_chunk"        # IL_GRASP_ACTION_CHUNK_TOPIC
ACTION_DIM = 7                                # dx dy dz, rx ry rz (XYZ Euler), gripper
TCP_TOPIC = "/left_arm/tcp_pose_world"       # link6 pose, world frame (sim_observation_publisher.py)
EE_LINK = "gripper_center"                    # END_EFFECTOR_FRAME_NAME: the point the IL actions move
# Sim world <- left arm base rotation, from Scene-MCHD-20.usd (/world/WorkSpace/RS_M90E7A_Left)
SIM_BASE_QUAT_XYZW = [-0.6118, -0.3545, 0.3538, 0.6122]


# ---------------- kinematics (URDF) ----------------
class UrdfChain:
    """Forward kinematics from the URDF: base_link -> any link, joints named Joint_1..Joint_6."""

    def __init__(self, path):
        self.joints = {}
        for j in ET.parse(path).getroot().iter("joint"):
            o, a = j.find("origin"), j.find("axis")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = Rot.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
            self.joints[j.find("child").get("link")] = (j.find("parent").get("link"), j.get("name"), j.get("type"), T, axis)

    def fk(self, link, q, stop_at="base_link"):
        T = np.eye(4)
        while link != stop_at:
            parent, name, typ, T_origin, axis = self.joints[link]
            T_joint = np.eye(4)
            if typ == "revolute":
                T_joint[:3, :3] = Rot.from_rotvec(axis * q[int(name.split("_")[1]) - 1]).as_matrix()
            T = T_origin @ T_joint @ T
            link = parent
        return T


# ---------------- dataset ----------------
def load_episode(root, episode):
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    table = pads.dataset(files).to_table(columns=["frame_index", "observation.state", "action"],
                                         filter=pads.field("episode_index") == episode)
    if table.num_rows == 0:
        raise SystemExit(f"episode {episode} not found (dataset has {info['total_episodes']} episodes)")
    order = np.argsort(table.column("frame_index").to_numpy())
    state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))[order]
    action = np.stack(table.column("action").to_numpy(zero_copy_only=False))[order].astype(np.float64)
    return info, state, action


class Replayer(Node):
    def __init__(self, actions, args):
        super().__init__("episode_replayer")
        self.args = args
        self.close_index = None
        self.actions = actions.copy()
        if args.max_actions:
            self.actions = self.actions[: args.max_actions]
        if args.gripper == "open":
            self.actions[:, 6] = 1.0            # keep the sim's IL step running for the whole replay
        else:
            closes = np.where(self.actions[:, 6] < 0.5)[0]
            if len(closes):                     # the sim ends its IL step at the first close: stop there
                self.actions = self.actions[: closes[0] + 1]
                self.close_index = int(closes[0])
        # What goes on the wire: DeltaActionPolicy reads [3:6] as XYZ Euler angles, the dataset holds
        # rotation vectors. The analysis keeps using self.actions (dataset format).
        self.wire = self.actions.copy()
        self.wire[:, 3:6] = Rot.from_rotvec(self.actions[:, 3:6]).as_euler("xyz")
        self.pub = self.create_publisher(Float64MultiArray, CHUNK_TOPIC, 10)
        self.create_subscription(PoseStamped, TCP_TOPIC, self._on_tcp, 10)
        self.create_subscription(UInt32, REQUEST_TOPIC, self._on_request, 10)
        self.create_timer(0.5, self._check_settled)
        self.latest = None
        self.started = None
        self.sent = 0                           # actions published so far
        self.applied = 0                        # estimated actions the sim has executed (see _on_tcp)
        self.answered = set()                   # request numbers already answered
        self.rows = []                          # (t, actions applied so far, link6 pos, link6 quat xyzw)
        self.done_at = None

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        if self.started is None:
            return
        # record from the start of the replay until `settle` seconds after the last action
        if self.done_at is None or time.time() - self.done_at < self.args.settle:
            self.rows.append((time.time() - self.started, self.applied, *self.latest))
        # The TCP pose is published once per physics step, before that step's task update, which executes
        # one queued action if the sim has one: so count one more applied action per message while chunks
        # we sent are still outstanding. (A step that only requests the next chunk executes nothing.)
        if self.applied < self.sent:
            self.applied += 1
        if self.applied >= len(self.actions) and self.done_at is None:
            self.done_at = time.time()
            self.get_logger().info(f"all {len(self.actions)} actions executed; recording {self.args.settle:.0f} s more for the arm to settle")

    def _on_request(self, msg):
        seq = int(msg.data)
        if seq in self.answered:                # the sim re-requests after 2 s without an answer; never resend old data
            return
        if self.started is None:
            others = [i for i in self.get_publishers_info_by_topic(CHUNK_TOPIC) if i.node_name != self.get_name()]
            if others:
                self.get_logger().error(f"another node publishes {CHUNK_TOPIC} ({others[0].node_name}); stop it first")
                raise SystemExit(1)
            if self.latest is None:
                self.get_logger().error(f"no {TCP_TOPIC} yet: is 'Publish Policy Observations' on and the sim playing?")
                raise SystemExit(1)
            self.started = time.time()
            self.rows.append((0.0, 0, *self.latest))
            self.get_logger().info(f"first chunk request (seq={seq}): replaying {len(self.actions)} actions "
                                   f"in chunks of {self.args.chunk_size}")
        if self.sent >= len(self.actions):
            if self.args.gripper == "open":
                self.get_logger().info(f"request seq={seq} after the last action: not answered (the sim holds its last target)",
                                       throttle_duration_sec=5.0)
            return
        chunk = self.wire[self.sent: self.sent + self.args.chunk_size]
        out = Float64MultiArray()
        out.layout.dim = [
            MultiArrayDimension(label=f"seq={seq}", size=len(chunk), stride=chunk.size),
            MultiArrayDimension(label="action", size=ACTION_DIM, stride=ACTION_DIM),
        ]
        out.data = chunk.ravel().tolist()
        self.pub.publish(out)
        self.answered.add(seq)
        self.sent += len(chunk)

    def _check_settled(self):
        if self.done_at is not None and time.time() - self.done_at > self.args.settle:
            raise SystemExit(0)


# ---------------- analysis ----------------
def analyse(chain, state, actions, rows, out):
    n = len(actions)
    R_wb = Rot.from_quat(SIM_BASE_QUAT_XYZW)
    # recorded: gripper_center pose after each action k (k = 0..n), world-frame rotation
    T_rec = np.stack([chain.fk(EE_LINK, q) for q in state[: n + 1, :6]])
    rec_p = R_wb.apply(T_rec[:, :3, 3])
    rec_r = R_wb * Rot.from_matrix(T_rec[:, :3, :3])
    rec_d = (rec_p - rec_p[0]) * 1000
    rec_ang = np.degrees((rec_r * rec_r[0].inv()).magnitude())
    # the actions integrated on their own: what perfect tracking would give
    act_d = np.vstack([np.zeros(3), np.cumsum(actions[:, :3], axis=0)]) * 1000
    # sim: link6 pose -> gripper_center (fixed offset from the URDF)
    T_l6_ee = chain.fk(EE_LINK, np.zeros(6), stop_at="link6")
    t = np.array([r[0] for r in rows]); k = np.array([r[1] for r in rows])
    l6p = np.array([r[2] for r in rows]); l6r = Rot.from_quat(np.array([r[3] for r in rows]))
    sim_p = l6p + l6r.apply(T_l6_ee[:3, 3])
    sim_r = l6r * Rot.from_matrix(T_l6_ee[:3, :3])
    sim_d = (sim_p - sim_p[0]) * 1000
    sim_ang = np.degrees((sim_r * sim_r[0].inv()).magnitude())
    rec_at = rec_d[np.minimum(k, n)]
    err = np.linalg.norm(sim_d - rec_at, axis=1)
    # "final" = when the last action had been executed: after a close the sim's scripted grasp moves the arm
    # (lift, retreat), which is not part of the replay; that motion is reported separately.
    last = int(np.argmax(k >= n)) if (k >= n).any() else len(k) - 1
    moved_after = float(np.linalg.norm(sim_d[-1] - sim_d[last]))
    sim_d_rep, err_rep = sim_d[: last + 1], err[: last + 1]
    fin_r, fin_s = rec_d[n], sim_d[last]
    path_r = np.linalg.norm(np.diff(rec_d, axis=0), axis=1).sum()
    path_s = np.linalg.norm(np.diff(sim_d_rep, axis=0), axis=1).sum()
    cosang = float(fin_r @ fin_s / (np.linalg.norm(fin_r) * np.linalg.norm(fin_s) + 1e-12))
    summary = {
        "actions_applied": int(k.max()), "replay_seconds": round(float(t[last]), 2),
        "final_displacement_recorded_mm": np.round(fin_r, 2).tolist(),
        "final_displacement_actions_mm": np.round(act_d[n], 2).tolist(),
        "final_displacement_sim_mm": np.round(fin_s, 2).tolist(),
        "final_error_sim_vs_recorded_mm": round(float(np.linalg.norm(fin_s - fin_r)), 2),
        "final_error_sim_vs_actions_mm": round(float(np.linalg.norm(fin_s - act_d[n])), 2),
        "net_motion_ratio_sim_over_recorded": round(float(np.linalg.norm(fin_s) / (np.linalg.norm(fin_r) + 1e-12)), 3),
        "path_length_recorded_mm": round(float(path_r), 1), "path_length_sim_mm": round(float(path_s), 1),
        "direction_agreement_deg": round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 2),
        "tracking_error_mean_mm": round(float(err_rep.mean()), 2), "tracking_error_max_mm": round(float(err_rep.max()), 2),
        "final_rotation_recorded_deg": round(float(rec_ang[n]), 2), "final_rotation_sim_deg": round(float(sim_ang[last]), 2),
        "final_rotation_error_deg": round(float(np.degrees((sim_r[last] * sim_r[0].inv() * (rec_r[n] * rec_r[0].inv()).inv()).magnitude())), 2),
        "moved_after_last_action_mm": round(moved_after, 1),
    }
    with open(out + ".csv", "w") as f:
        f.write("t_s,actions_applied,sim_dx_mm,sim_dy_mm,sim_dz_mm,sim_rot_deg,rec_dx_mm,rec_dy_mm,rec_dz_mm,rec_rot_deg,error_mm\n")
        for i in range(len(t)):
            kk = min(k[i], n)
            f.write(f"{t[i]:.3f},{k[i]},{sim_d[i,0]:.3f},{sim_d[i,1]:.3f},{sim_d[i,2]:.3f},{sim_ang[i]:.3f},"
                    f"{rec_d[kk,0]:.3f},{rec_d[kk,1]:.3f},{rec_d[kk,2]:.3f},{rec_ang[kk]:.3f},{err[i]:.3f}\n")
    write_report(out + ".html", summary, k, sim_d, sim_ang, rec_d, rec_ang, n)
    return summary


def write_report(path, summary, k, sim_d, sim_ang, rec_d, rec_ang, n):
    def panel(title, unit, rec_y, sim_x, sim_y):
        W, H, L, R, T, B = 460, 210, 48, 12, 10, 26
        xs = np.arange(len(rec_y)); lo = min(rec_y.min(), sim_y.min()); hi = max(rec_y.max(), sim_y.max())
        pad = (hi - lo) * 0.08 or 1.0; lo -= pad; hi += pad
        X = lambda v: L + v / max(n, 1) * (W - L - R)
        Y = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - B)

        def line(x, y, cls):
            step = max(1, len(x) // 600)
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x[::step], y[::step]))
            return f'<polyline class="{cls}" points="{pts}"/>'
        ticks = np.linspace(lo, hi, 5)
        grid = "".join(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                       f'<text class="tick" x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>' for v in ticks)
        xt = "".join(f'<text class="tick" x="{X(v):.1f}" y="{H-8}" text-anchor="middle">{int(v)}</text>' for v in np.linspace(0, n, 5))
        return (f'<figure><figcaption><b>{html.escape(title)}</b> <span>({unit})</span></figcaption>'
                f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{grid}{xt}'
                f'{line(xs, rec_y, "rec")}{line(sim_x, sim_y, "sim")}</svg></figure>')
    panels = "".join(panel(f"{ax} displacement from start", "mm, world frame", rec_d[:, i], k, sim_d[:, i]) for i, ax in enumerate("xyz"))
    panels += panel("rotation from start", "deg", rec_ang, k, sim_ang)
    rows = "".join(f"<tr><td>{html.escape(a.replace('_', ' '))}</td><td>{html.escape(json.dumps(b))}</td></tr>" for a, b in summary.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Episode Replay</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9; --rec:#2a78d6; --sim:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --grid:#2c2c2a; --rec:#3987e5; --sim:#d95926; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:20px 16px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:12px; }}
figure {{ margin:0; background:var(--surface); border:1px solid rgba(127,127,127,.2); border-radius:10px; padding:10px 12px; }}
figcaption span, .legend {{ color:var(--ink2); }} svg {{ width:100%; height:auto; }}
.grid {{ stroke:var(--grid); }} .tick {{ fill:var(--ink2); font-size:10px; }}
.rec, .sim {{ fill:none; stroke-width:2; stroke-linejoin:round; }} .rec {{ stroke:var(--rec); }} .sim {{ stroke:var(--sim); }}
.legend i {{ display:inline-block; width:14px; height:2px; vertical-align:middle; margin:0 6px 0 14px; }}
table {{ border-collapse:collapse; margin-top:12px; font-variant-numeric:tabular-nums; }} td {{ padding:3px 12px 3px 0; border-bottom:1px solid var(--grid); }}
</style></head><body><main><h1 style="font-size:20px">Episode Replay: Isaac Sim vs recorded episode</h1>
<p class="legend"><i style="background:var(--rec)"></i>recorded episode<i style="background:var(--sim)"></i>Isaac Sim (replayed). x axis: actions executed by the sim.</p>
<div class="grid2">{panels}</div><table>{rows}</table></main></body></html>"""
    with open(path, "w") as f:
        f.write(doc)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("REPLAY_DATASET", "/dataset"), help="LeRobot dataset root")
    ap.add_argument("--urdf", default=os.environ.get("REPLAY_URDF", "/urdf/RS_M90E7A_MCHD_20.urdf"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8,
                    help="actions per chunk (default 8 = n_action_steps of the fan-grasp diffusion policies, which "
                         "ros2_policy_inference sends as its chunk size); the sim runs one idle step per chunk while "
                         "it requests the next, as with a real policy")
    ap.add_argument("--gripper", choices=["recorded", "open"], default="recorded",
                    help="recorded: send the episode's gripper and stop at its first close, where the sim ends its "
                         "IL step and runs its scripted grasp (default); open: send gripper=1 throughout so the "
                         "whole episode, including the lift after the close, is replayed")
    ap.add_argument("--max-actions", type=int, default=0, help="replay only the first N actions (0 = all)")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to keep recording after the last action")
    ap.add_argument("--out", default=None, help="output path without extension (default: results/ep<N>_<time>)")
    args = ap.parse_args()
    if args.chunk_size < 1:
        ap.error("--chunk-size must be at least 1")

    info, state, actions = load_episode(args.dataset, args.episode)
    chain = UrdfChain(args.urdf)
    if args.out is None:
        args.out = os.path.join(here, "results", f"ep{args.episode:03d}_{args.gripper}_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rclpy.init()
    node = Replayer(actions, args)
    if node.close_index is not None:
        node.get_logger().info(f"--gripper recorded: replaying up to and including the gripper close at action {node.close_index}")
    node.get_logger().info(f"episode {args.episode}: {len(node.actions)} of {len(actions)} actions from {args.dataset}; "
                           f"waiting for the first request on {REQUEST_TOPIC}: start the task in the Delta Extension panel")
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):
        pass
    rows, replayed = node.rows, node.actions
    node.destroy_node()
    rclpy.try_shutdown()
    if len(rows) < 10:
        print("too few TCP samples recorded; nothing to compare")
        return
    summary = analyse(chain, state, replayed, rows, args.out)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}.csv and {args.out}.html")


if __name__ == "__main__":
    main()
