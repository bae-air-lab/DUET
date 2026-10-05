#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <algorithm>
#include <cmath>
#include <mutex>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Runaway-policy guard: drop to Passive (damping) if the raw policy
    // output stays above guard_max_abs_action_ for guard_hold_s_. Time-based
    // because the checks run at the FSM rate, not the 50 Hz policy rate.
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    if (deploy_cfg["safety"]) {
        auto s = deploy_cfg["safety"];
        if (s["max_abs_action"]) guard_max_abs_action_ = s["max_abs_action"].as<float>();
        if (s["max_abs_action_hold_s"]) guard_hold_s_ = s["max_abs_action_hold_s"].as<float>();
    }
    spdlog::info("RL guard: Passive if |raw action| > {:.1f} for {:.0f} ms",
                 guard_max_abs_action_, 1000.0f * guard_hold_s_);
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return action_runaway(); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Subscribe to upper-body (arm) targets published by the IL / GR00T policy on
    // a dedicated topic (reuses the LowCmd_ IDL, so no new message type). If no
    // message arrives within the timeout the arms fall back to the carry pose.
    arm_sub_ = std::make_shared<unitree::robot::g1::subscription::LowCmd>("rt/arm_targets");
    arm_sub_->set_timeout_ms(200);
}

bool State_RLBase::action_runaway()
{
    const auto raw = env->action_manager->action();
    float peak = 0.0f;
    for (float v : raw) peak = std::max(peak, std::fabs(v));

    if (peak <= guard_max_abs_action_) {
        guard_violating_ = false;
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!guard_violating_) {
        guard_violating_ = true;
        guard_since_ = now;
        return false;
    }
    if (std::chrono::duration<float>(now - guard_since_).count() < guard_hold_s_) {
        return false;
    }
    spdlog::error("RL guard: |raw action| {:.1f} > {:.1f} for {:.0f} ms -> Passive",
                  peak, guard_max_abs_action_, 1000.0f * guard_hold_s_);
    guard_violating_ = false;
    return true;
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    auto & jmap = env->robot->data.joint_ids_map;
    auto & qdef = env->robot->data.default_joint_pos;
    const int n = (int)jmap.size();
    const int n_lower = (int)action.size();
    const int n_arm = n - n_lower;

    // One-time init of the ramp buffer to the default (carry) arm pose.
    if (!arm_ramp_init_) {
        arm_q_applied_.assign(n_arm, 0.0f);
        for (int k = 0; k < n_arm; ++k) arm_q_applied_[k] = (float)qdef[n_lower + k];
        arm_ramp_init_ = true;
    }

    // run() is called at ~1 kHz (CtrlFSM), so use a wall-clock dt for the ramp.
    const auto now = std::chrono::steady_clock::now();
    const float dt = arm_first_run_
        ? (float)env->step_dt
        : std::chrono::duration<float>(now - arm_last_tp_).count();
    arm_last_tp_ = now;
    arm_first_run_ = false;
    // "Active" = an IL (GR00T) message arrived within the subscriber timeout.
    const bool arm_active = !arm_sub_->isTimeout();

    // Remote arm-pose selection via the gamepad (imported from arm.py). A live
    // rt/arm_targets (IL/GR00T) stream still overrides these when present.
    //   Y : toggle DEFAULT (carry) <-> FORWARD (arms out to receive a box)
    //   B : toggle FORWARD <-> BOX_HOLD (arms in, gripping) [only in those states]
    {
        auto js = env->robot->data.joystick;
        const bool y_now = js->Y.pressed;
        const bool b_now = js->B.pressed;
        if (y_now && !arm_prev_y_) {
            arm_pose_ = (arm_pose_ == ArmPose::DEFAULT) ? ArmPose::FORWARD
                                                        : ArmPose::DEFAULT;
        }
        if (b_now && !arm_prev_b_) {
            if      (arm_pose_ == ArmPose::FORWARD)  arm_pose_ = ArmPose::BOX_HOLD;
            else if (arm_pose_ == ArmPose::BOX_HOLD) arm_pose_ = ArmPose::FORWARD;
        }
        arm_prev_y_ = y_now;
        arm_prev_b_ = b_now;
    }

    // Arm-pose presets (imported from arm.py), mapped to the G1's 10 arm joints in
    // jmap arm order [L: sh_pitch, sh_roll, sh_yaw, elbow, wrist_roll, R: ...];
    // wrist_roll (idx 4/9) is 0. Applied only when the arm layout is the expected
    // 10 joints, else fall back to the carry/default pose.
    static const float POSE_FORWARD[10] = {
        -1.35f, 0.50f, 0.0f, 0.87f, 0.0f,  -1.35f, -0.50f, 0.0f, 0.87f, 0.0f};
    static const float POSE_BOXHOLD[10] = {
        -1.25f, 0.00f, 0.0f, 0.70f, 0.0f,  -1.25f,  0.00f, 0.0f, 0.70f, 0.0f};
    const bool pose_ok = (n_arm == 10);

    // Ramp rate: fast+transparent while tracking IL; gentle when homing to the
    // default/carry pose (finger-safe); moderate for a deliberate button pose.
    const float ARM_RATE_ACTIVE = 20.0f;  // rad/s, transparent tracking of IL
    const float ARM_RATE_RETURN = 0.7f;   // rad/s, gentle homing to default/carry
    const float ARM_RATE_POSE   = 2.0f;   // rad/s, deliberate button-selected pose
    float arm_rate;
    if (arm_active)                         arm_rate = ARM_RATE_ACTIVE;
    else if (arm_pose_ == ArmPose::DEFAULT) arm_rate = ARM_RATE_RETURN;
    else                                    arm_rate = ARM_RATE_POSE;
    const float max_step = arm_rate * dt;

    // Snapshot the IL arm command once (the DDS callback writes msg_ on another
    // thread).
    std::vector<float> il_q(n_arm), il_kp(n_arm), il_kd(n_arm), il_tau(n_arm);
    if (arm_active) {
        std::lock_guard<std::mutex> lock(arm_sub_->mutex_);
        for (int k = 0; k < n_arm; ++k) {
            const int motor = (int)jmap[n_lower + k];
            il_q[k]   = arm_sub_->msg_.motor_cmd()[motor].q();
            il_kp[k]  = arm_sub_->msg_.motor_cmd()[motor].kp();
            il_kd[k]  = arm_sub_->msg_.motor_cmd()[motor].kd();
            il_tau[k] = arm_sub_->msg_.motor_cmd()[motor].tau();
        }
    }

    for (int i = 0; i < n; ++i) {
        const int motor = (int)jmap[i];
        if (i < n_lower) {
            // Lower body: RL policy.
            lowcmd->msg_.motor_cmd()[motor].q() = action[i];
            continue;
        }
        // Arm joint: follow IL target when live, else the button-selected pose
        // (FORWARD / BOX_HOLD), else the default carry pose.
        const int k = i - n_lower;
        float target;
        if (arm_active)                                     target = il_q[k];
        else if (pose_ok && arm_pose_ == ArmPose::FORWARD)  target = POSE_FORWARD[k];
        else if (pose_ok && arm_pose_ == ArmPose::BOX_HOLD) target = POSE_BOXHOLD[k];
        else                                                target = (float)qdef[i];
        const float err = target - arm_q_applied_[k];
        if (std::fabs(err) <= max_step) arm_q_applied_[k] = target;
        else arm_q_applied_[k] += std::copysign(max_step, err);

        lowcmd->msg_.motor_cmd()[motor].q() = arm_q_applied_[k];
        lowcmd->msg_.motor_cmd()[motor].dq() = 0;
        if (arm_active) {
            lowcmd->msg_.motor_cmd()[motor].kp() = il_kp[k];
            lowcmd->msg_.motor_cmd()[motor].kd() = il_kd[k];
            lowcmd->msg_.motor_cmd()[motor].tau() = il_tau[k];
        } else {
            lowcmd->msg_.motor_cmd()[motor].tau() = 0;
        }
    }
}