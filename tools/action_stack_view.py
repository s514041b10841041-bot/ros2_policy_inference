#!/usr/bin/env python3
"""Live view of the policy's action chunks, stacked by sim step, for a terminal of its own.

It only listens to the topics the sim and the inference node already use, so it changes nothing in
either. Start it with tools/action_stack_view.sh, before or after the inference node.

Each column is one sim step and each cell a chunk's gripper command at that step, as a digit from
0 (closed) to 9 (open). A '*' marks the actions the sim really ran, so the view shows where chunks
overlap, which rows were dropped, and which chunk a close came from:

    action stack at step 215: gripper 0=closed 9=open, *=executed
      step    178       183       188       193       198       203       208       213
    #19 @178  9 9 9*9*9*9*9*9*9*9*9*9*9*9*9*
    #20 @186                  9 9 9 9 9 9 9 9*9*9*9*9*9*9*7*
    #21 @200                                              9 9*9*9*9*9*9*9*9*9*9*9*9*9*0*

Each new grasp starts a new stack below the last one, so earlier grasps stay in the scrollback.
"""
import argparse
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, UInt32MultiArray

ACTION_DIM = 7
GRIPPER = 6  # action column of the gripper command


class ActionStack:
    """Keeps the last `keep` chunks. Plain chunks have no start step: they run back to back, whole."""

    def __init__(self, keep=6):
        self._chunks = deque(maxlen=keep)
        self._next_start = 0

    def clear(self):
        self._chunks.clear()
        self._next_start = 0

    def add(self, seq, actions, start=None):
        """Record a chunk. seq 1 is a new grasp. start: the sim step of row 0 (None = plain chunk)."""
        if seq == 1:
            self.clear()
        gripper = np.asarray(actions)[:, GRIPPER]
        first = self._next_start if start is None else start
        self._chunks.append({"seq": seq, "start": first, "gripper": gripper,
                             "begin": first if start is None else None})
        self._next_start = first + len(gripper)

    def installed(self, seq, skipped):
        """The sim started running chunk `seq`, after dropping its first `skipped` actions (RTC)."""
        for chunk in self._chunks:
            if chunk["seq"] == seq:
                chunk["begin"] = chunk["start"] + skipped

    def render(self, step=None):
        """The stack as text. step: actions the sim has executed (None = up to the newest chunk)."""
        chunks = list(self._chunks)
        last_step = max(c["start"] + len(c["gripper"]) for c in chunks)
        step = chunks[-1]["start"] if step is None else step
        first_step = min(c["start"] for c in chunks)
        lines = [f"action stack at step {step}: gripper 0=closed 9=open, *=executed",
                 "  step    " + "".join(f"{s:<10}" for s in range(first_step, last_step, 5))]
        for i, chunk in enumerate(chunks):
            # a chunk runs until the next installed chunk replaces it
            begins = [c["begin"] for c in chunks[i + 1:] if c["begin"] is not None]
            stop = min(begins[0] if begins else step, step)
            cells = ["  "] * (last_step - first_step)
            for row, value in enumerate(chunk["gripper"]):
                at = chunk["start"] + row
                ran = chunk["begin"] is not None and chunk["begin"] <= at < stop
                cells[at - first_step] = f"{int(np.clip(value, 0.0, 0.999) * 10)}{'*' if ran else ' '}"
            lines.append(f"#{chunk['seq']:<3}@{chunk['start']:<5}" + "".join(cells).rstrip())
        return "\n".join(lines)


class ActionStackView(Node):
    def __init__(self, namespace):
        super().__init__("action_stack_view")
        self._stack = ActionStack()
        self._step = None      # actions the sim has executed; None until it publishes (RTC only)
        self._drawn_lines = 0  # lines of the current stack on screen, redrawn in place
        self._dirty = False
        self.create_subscription(Float64MultiArray, f"{namespace}/action_chunk", self._on_chunk, 10)
        self.create_subscription(UInt32MultiArray, f"{namespace}/exec_step", self._on_exec_step, 50)
        self.create_timer(0.1, self._draw)
        print(f"waiting for chunks on {namespace}/action_chunk ...", flush=True)

    def _on_chunk(self, msg):
        label = msg.layout.dim[0].label if msg.layout.dim else ""
        fields = dict(f.split("=", 1) for f in label.split(";") if "=" in f)
        data = np.asarray(msg.data, dtype=np.float64)
        if "seq" not in fields or data.size == 0 or data.size % ACTION_DIM:
            return
        seq = int(fields["seq"])
        start = int(fields["start"]) if "start" in fields else None
        if seq == 1:
            self._next_grasp()
        self._stack.add(seq, data.reshape(-1, ACTION_DIM), start=start)
        self._dirty = True

    def _on_exec_step(self, msg):
        if len(msg.data) < 3:
            return
        step, installed_seq, dropped = (int(v) for v in msg.data[:3])
        if self._step is not None and step < self._step:  # the sim's counter restarts with each grasp
            self._next_grasp()
        self._step = step
        if installed_seq:
            self._stack.installed(installed_seq, dropped)
        self._dirty = True

    def _next_grasp(self):
        """Leave the last grasp's final stack on screen and start a new one below it."""
        self._draw()
        self._drawn_lines = 0
        self._stack.clear()
        self._step = None

    def _draw(self):
        if not self._dirty or not self._stack._chunks:
            return
        self._dirty = False
        text = self._stack.render(self._step)
        up = f"\033[{self._drawn_lines}F\033[J" if self._drawn_lines else ""
        print(up + text, flush=True)
        self._drawn_lines = text.count("\n") + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--namespace", default="/il/grasp", help="topic prefix (default: /il/grasp)")
    args = parser.parse_args()
    rclpy.init()
    node = ActionStackView(args.namespace)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
