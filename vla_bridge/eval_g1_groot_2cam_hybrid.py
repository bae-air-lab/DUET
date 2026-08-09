"""
GR00T eval client for Unitree G1_23 — TWO CAMERAS (head + wrist).
HYBRID build: arms only, driven alongside the RL lower-body controller.

Difference vs eval_g1_groot_2cam.py (standalone):
    This version publishes the arm LowCmd to "rt/arm_targets" instead of "rt/lowcmd"
    (via G1_23_ArmController(lowcmd_topic="rt/arm_targets")). The RL controller
    (g1_ctrl) is the SOLE rt/lowcmd writer: it runs the legs + waist and MERGES
    these arm targets onto the 10 arm joints. So GR00T must NOT touch rt/lowcmd
    here, or it would fight g1_ctrl. The Dex3 hands are on their own topic and are
    unaffected. Run g1_ctrl FIRST (it owns rt/lowcmd), then this.

Sends both cameras to the policy:
    cam_left_high   <- head frame   (was color_0 in training data)
    cam_right_wrist <- wrist frame  (was color_2 in training data)

Control path (PROVEN by replay + official Unitree eval_g1.py):
    motion_mode=False  +  tau = arm_ik.solve_tau(arm_action)
    arm_ctrl.ctrl_dual_arm(arm_action, tau)

MUST be run from ~/unitree_lerobot  (G1_23_ArmIK loads a RELATIVE-path URDF).
Run in the `unitree_lerobot` conda env (working pinocchio + PolicyClient guard).
Policy server (Isaac-GR00T .venv) runs separately on :5555.
"""
import fix_logging_mp
import numpy as np
import time
import argparse
import cv2

from gr00t.policy.server_client import PolicyClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_23_ArmController
from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import Dex3_1_Controller
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_23_ArmIK

from multiprocessing import Array


def recursive_add_extra_dim(obs):
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class G1GrootAdapter:
    def __init__(self, policy_client):
        self.policy = policy_client

    def get_action(self, head_img, wrist_img, state, lang):
        # both imgs: (480, 640, 3) RGB, matching training keys
        model_obs = {
            "video": {
                "cam_left_high":   head_img,
                "cam_right_wrist": wrist_img,
            },
            "state": {
                "left_arm":   state[:5].astype(np.float32),
                "right_arm":  state[5:10].astype(np.float32),
                "left_hand":  state[10:17].astype(np.float32),
                "right_hand": state[17:24].astype(np.float32),
            },
            "language": {
                "annotation.human.task_description": lang,
            },
        }
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        action_chunk, info = self.policy.get_action(model_obs)
        horizon = action_chunk[next(iter(action_chunk))].shape[1]
        actions = []
        for t in range(horizon):
            actions.append(np.concatenate([
                action_chunk["left_arm"][0][t],
                action_chunk["right_arm"][0][t],
                action_chunk["left_hand"][0][t],
                action_chunk["right_hand"][0][t],
            ]))
        return actions


def _to_rgb_480_640(frame, want_rgb=True):
    """request_bgr=True -> .bgr is BGR; convert to RGB (LeRobot stores RGB)."""
    if want_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape[:2] != (480, 640):
        frame = cv2.resize(frame, (640, 480))  # cv2 size is (W,H)
    return np.ascontiguousarray(frame)


def get_head_frame(img_client, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hf = img_client.get_head_frame()
        if hf is not None and hf.bgr is not None:
            return _to_rgb_480_640(hf.bgr)
        time.sleep(0.01)
    return None


def get_wrist_frame(img_client, timeout=2.0):
    """color_2 in training was the WRIST cam. Verify which physical wrist this is:
    if your wrist camera is the RIGHT wrist mount use get_right_wrist_frame();
    if it's the LEFT wrist mount, switch to get_left_wrist_frame() below.
    The frame MUST come from the same physical camera that was color_2 in training."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        wf = img_client.get_right_wrist_frame()   # <-- swap to get_left_wrist_frame() if needed
        if wf is not None and wf.bgr is not None:
            return _to_rgb_480_640(wf.bgr)
        time.sleep(0.01)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", default="localhost")
    parser.add_argument("--policy_port", type=int, default=5555)
    parser.add_argument("--img_server_ip", default="192.168.123.164")
    parser.add_argument("--img_request_port", type=int, default=60000)
    parser.add_argument("--frequency", type=float, default=15.0)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--lang", default="pick up cube and place it in the tray.")
    parser.add_argument("--network_interface", default="enp132s0")
    parser.add_argument("--smooth_alpha", type=float, default=1.0,
                        help="1.0 = no smoothing; lower (e.g. 0.7) low-pass filters arm cmds")
    parser.add_argument("--force-limit", type=float, default=2.0,
                        help="Soft grasp limit: clip hand q to ±this value (rad). Lower = gentler. Default 2.0 (no limit)")
    args = parser.parse_args()

    # Initialize DDS
    ChannelFactoryInitialize(0, args.network_interface)

    # Robot controllers  (motion_mode=False is the PROVEN config)
    # HYBRID: publish arm targets to rt/arm_targets so the RL controller (g1_ctrl,
    # the sole rt/lowcmd writer) merges them onto the arm joints while it runs the
    # legs. Set lowcmd_topic back to "rt/lowcmd" to drive the robot standalone.
    arm_ctrl = G1_23_ArmController(motion_mode=False, lowcmd_topic="rt/arm_targets")
    arm_ik = G1_23_ArmIK()

    # Two-step pre-pose: lift shoulder first (roll only), then bring arm forward.
    # This avoids the arm sweeping through the hip on the way to the ready pose.
    # Joint order: [L Pitch, L Roll, L Yaw, L Elbow, L WristRoll,
    #               R Pitch, R Roll, R Yaw, R Elbow, R WristRoll]
    SHOULDER_ROLL = 0.75  # ~43 deg

    # Step 1: only open ShoulderRoll, keep all other joints at their current values
    step1_pose = arm_ctrl.get_current_dual_arm_q().copy()
    step1_pose[1] = SHOULDER_ROLL    # L ShoulderRoll
    step1_pose[6] = -SHOULDER_ROLL   # R ShoulderRoll
    arm_ctrl.ctrl_dual_arm(step1_pose, np.zeros(10))
    time.sleep(2.0)

    # Step 2: bring arm to forward position with shoulder open
    ready_pose = np.array([0.0, SHOULDER_ROLL, 0.0, 0.0, 0.0,
                           0.0, -SHOULDER_ROLL, 0.0, 0.0, 0.0])
    arm_ctrl.ctrl_dual_arm(ready_pose, np.zeros(10))
    time.sleep(2.0)

    left_hand_array = Array("d", 7)
    right_hand_array = Array("d", 7)
    hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array)
    time.sleep(1.0)

    # Camera (request/response API)
    img_client = ImageClient(
        host=args.img_server_ip,
        request_port=args.img_request_port,
        request_bgr=True,
    )
    cfg = img_client.get_cam_config()
    hc = cfg["head_camera"]
    print(f"[cam] head image_shape={hc.get('image_shape')} binocular={hc.get('binocular')}")
    assert not hc.get("binocular", False), "Head camera is binocular! Trained on mono — fix server."

    # Confirm BOTH frames flow before starting
    th = get_head_frame(img_client)
    tw = get_wrist_frame(img_client)
    if th is None:
        raise RuntimeError("No HEAD frames from image server. Is image_server running?")
    if tw is None:
        raise RuntimeError("No WRIST frames. Check wrist camera + get_*_wrist_frame() choice.")
    print(f"[cam] head OK {th.shape} {th.dtype} | wrist OK {tw.shape} {tw.dtype}")

    # GR00T policy
    policy_client = PolicyClient(host=args.policy_host, port=args.policy_port)
    adapter = G1GrootAdapter(policy_client)

    print("Press Enter to start, Ctrl+C to stop.")
    input()

    action_queue = []
    smoothed = None
    step = 0
    try:
        while True:
            loop_start = time.perf_counter()

            head_img  = get_head_frame(img_client)
            wrist_img = get_wrist_frame(img_client)
            if head_img is None or wrist_img is None:
                print("Frame drop (head or wrist), skipping step")
                time.sleep(1.0 / args.frequency)
                continue

            # State
            arm_q = arm_ctrl.get_current_dual_arm_q()
            hand_state = np.array(list(left_hand_array[:]) + list(right_hand_array[:]))
            state = np.concatenate([arm_q, hand_state])

            # Query policy when queue empty
            if len(action_queue) == 0:
                actions = adapter.get_action(head_img, wrist_img, state, args.lang)
                action_queue = actions[:args.action_horizon]
                print(f"Step {step}: got {len(action_queue)} actions")

            # Execute
            action = action_queue.pop(0)
            arm_action = action[:10]

            # optional low-pass smoothing on arm command
            if args.smooth_alpha < 1.0:
                if smoothed is None:
                    smoothed = arm_action.copy()
                smoothed = args.smooth_alpha * arm_action + (1 - args.smooth_alpha) * smoothed
                arm_cmd = smoothed
            else:
                arm_cmd = arm_action

            print("arm:", np.round(arm_cmd, 3), " hand:", np.round(action[10:24], 3))

            # gravity compensation (PROVEN: solve_tau on the commanded target)
            tau = arm_ik.solve_tau(arm_cmd)
            arm_ctrl.ctrl_dual_arm(arm_cmd, tau)

            # Apply soft grasp limit to prevent motor shutdown from over-closure
            hand_action = action[10:24].copy()
            hand_action = np.clip(hand_action, -args.force_limit, args.force_limit)

            left_hand_array[:] = hand_action[0:7].tolist()
            right_hand_array[:] = hand_action[7:14].tolist()

            step += 1
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, (1.0 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Two-step shutdown: bring arm to forward position first, then close shoulder.
        # Step 1: keep current pitch/yaw, only bring ShoulderRoll back to zero
        step1_close = arm_ctrl.get_current_dual_arm_q().copy()
        step1_close[1] = 0.0
        step1_close[6] = 0.0
        arm_ctrl.ctrl_dual_arm(step1_close, np.zeros(10))
        time.sleep(2.0)

        # Step 2: return fully to home (all zeros)
        arm_ctrl.ctrl_dual_arm(np.zeros(10), np.zeros(10))
        time.sleep(2.0)

        img_client.close()


if __name__ == "__main__":
    main()