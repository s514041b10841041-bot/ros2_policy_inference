# 🤖 ROS 2 Policy Inference

Run IL / VLA policies with camera and robot observations.
Publish predicted end-effector actions to `/il/grasp/action`.

## 🏗️ Architecture

![ROS 2 Policy Inference Docker architecture](architecture.svg)

Each policy's Python dependencies are isolated in its own Docker image, built from a
dedicated Dockerfile (see [Publish a Docker image](#publish-a-docker-image)). Every image
mounts the same source code and checkpoints at container start.
## 🔑 Setup

Get a GitLab token (scope: `read_api`) from your GitLab profile → Access Tokens, then save it once — scripts pick it up automatically:

```bash
printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
```

## 🚀 Run

> 💡 Start with `dummy` to check inputs and preprocessing without a checkpoint.

### 🐳 Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
./run_docker.sh --policy-name act
./run_docker.sh --policy-name act bash   # shell instead of launching
```

| Policy | Status |
|:--|:--|
| `dummy` | no-op placeholder |
| `act` | ACT LeRobot checkpoint, working |
| `pi` | π₀.₅ (pi05) LeRobot checkpoint, working |
| `diffusion` | Diffusion Policy LeRobot checkpoint, working |
| `groot` | GR00T N1.7 LeRobot checkpoint, working |
| `liteflow` | InternVL3-1B + FLOWER, working |
| `act_jepa` | ACT-JEPA export (`policies/act_jepa/config.yaml`), loads; see its known gripper issue |
| `sim_act_jepa_50k` / `_87_5k` / `_125k` | ACT-JEPA on Le-007-turbo (RGB + turbo depth, gripper hidden from the policy), loads |
| `le008_act` / `le008_diffusion` | ACT / Diffusion Policy on le_008_head (wrist RGB + head RGB + wrist depth, 10 fps), loads |
| `le008_act_jepa_50k` / `_35k` | ACT-JEPA on le_008_head, same inputs; 35k has the lowest validation loss, loads |
| `omnireset` | rl-template OmniReset student / RL fine-tune, CPU only, own image |

On the real-robot PC (not the sim-only PC), pin DDS to the 10G camera link by creating `docker/.env` (gitignored, repo root):
```bash
echo 'CYCLONEDDS_URI=file:///workspace/src/ros2_policy_inference/docker/cyclonedds_10g.xml' > docker/.env
```

Checkpoints hosted on Hugging Face (e.g. `pi`'s) are downloaded with:
```bash
hf download <hf_repo_id> --local-dir checkpoints/<name>
```

The `le008_*` policies need all their weights in `checkpoints/le008_policies/`:
```bash
hf download ArthurGoldFishKing/le008_policies --local-dir checkpoints/le008_policies
```
They also subscribe to the head camera (`head_rgb_topic` in their `config.yaml`, default
`/camera_H/color/image_raw/compressed`) and wait for it before answering a request. le_008_head was
recorded at 10 fps, so the consumer must execute one action every 0.1 s.

### 💻 Local

Requires ROS 2 Jazzy and the policy's dependencies.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
ros2 launch ros2_policy_inference inference.launch.py policy_name:=act
```

### 📡 Check output

```bash
ros2 topic echo /il/grasp/action
```

## 🔌 ROS Interface

### Inputs

| Topic | Message type | Preprocessed observation |
|:--|:--|:--|
| `/camera_L/color/image_raw/compressed` | `CompressedImage` | RGB: `256 × 256 × 3` |
| `/camera_L/depth/aligned_depth_to_color` | `Image` | Depth: `256 × 256`, nearest-neighbor resize |
| `/camera_H/color/image_raw/compressed` | `CompressedImage` | Head RGB: `256 × 256 × 3`; only for policies that set `head_rgb_topic` (`le008_*`) |
| `/left_arm/joint_states` | `JointState` | Position: `(6,)`; velocity: `(6,)` |
| `/left_arm/tcp_pose_world` | `PoseStamped` | EE-offset position: `(3,)`; quaternion: `(4,)` |
| `/left_ft/wrench_compensated` | `WrenchStamped` | `[fx, fy, fz, tx, ty, tz]` |

### Output

| Topic | Message type | Format |
|:--|:--|:--|
| `/il/grasp/action` | `Float64MultiArray` | `[dx, dy, dz, drx, dry, drz, gripper]` |

## 🛠️ Development

### Add a policy

1. Subclass `PolicyModel` in `policies/your_policy.py`; see `policies/dummy_policy.py`.
2. Define observation and action specs in `policies/base_policy.py`.
3. Register the policy in `policies/__init__.py` through `create_policy()`.

### Publish a Docker image

Rebuild and publish only when a Dockerfile or its dependencies change. Needs
a token with `api` scope (higher than the `read_api` one from Setup) in
`~/.gitlab_token` or `$GITLAB_TOKEN`:

```bash
./push_docker.sh             # act/pi/dummy image
./push_docker.sh --liteflow   # liteflow image
./push_docker.sh --omnireset  # omnireset image
```