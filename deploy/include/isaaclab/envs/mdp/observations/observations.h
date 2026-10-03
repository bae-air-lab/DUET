// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <Eigen/Geometry>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(base_height_command)
{
    // Target base (pelvis) height, driven by the remote D-pad:
    //   Up   = stand taller (ramp toward hi)
    //   Down = squat lower  (ramp toward lo)
    // The target integrates while a key is held and persists across control
    // steps (single-robot deploy), so the height stays where the user left it.
    auto & joystick = env->robot->data.joystick;
    const auto base_height_cfg = env->cfg["commands"]["base_height"];
    const auto cfg = base_height_cfg["range"];
    const float lo = cfg[0].as<float>();
    const float hi = cfg[1].as<float>();
    // Resting height when idle with no D-pad input. Defaults to a hair below the
    // 0.75 max so the standing pose sits just inside the trained band (off the
    // exact-max edge, slightly more stable) while hi=0.75 stays reachable via
    // D-pad Up. Falls back to hi if `standstill` is absent from the config.
    const float standstill = base_height_cfg["standstill"]
                                 ? base_height_cfg["standstill"].as<float>()
                                 : hi;

    static float target = standstill;               // start at the resting height
    const float ADJ_RATE = 0.40f;                   // m/s — how fast Up/Down move the target
    const float adj_step = ADJ_RATE * env->step_dt;
    if (joystick->up.pressed)   target += adj_step;
    if (joystick->down.pressed) target -= adj_step;
    if (joystick->left.pressed)  target = 0.60f;  // D-pad Left: walk-height preset (walk-band floor)
    if (joystick->right.pressed) target = hi;     // D-pad Right: full-stand preset (top of range, 0.73)
    target = std::clamp(target, lo, hi);

    // Velocity/height decoupling (Run-7 walk band): the policy learned to WALK at
    // any height in [WALK_MIN, hi] and to squat deeper only while standing still.
    // So while a locomotion command is active, clamp the target up to WALK_MIN --
    // the robot still walks at the user's commanded height (e.g. 0.62) but never
    // deep-folds while walking. The deep-squat target is remembered, resumes on stop.
    float cmd_target = target;
    const float WALK_MIN = 0.60f;  // matches BaseHeightCommandCfg.walk_min_height
    const float DEAD = 0.05f;
    if (std::fabs(joystick->ly()) > DEAD ||
        std::fabs(joystick->lx()) > DEAD ||
        std::fabs(joystick->rx()) > DEAD) {
        cmd_target = std::max(target, WALK_MIN);
    }

    // Velocity-limited ramp so the squat glides smoothly instead of jumping.
    // Persists across control steps. Tune RATE (m/s) for a faster/slower squat.
    static float h = standstill;
    const float RATE = 0.40f;
    const float max_step = RATE * env->step_dt;
    const float err = cmd_target - h;
    if (std::fabs(err) <= max_step) h = cmd_target;
    else h += std::copysign(max_step, err);

    // ACHIEVED pelvis height above the lower foot, printed next to the command
    // so the two can be compared. They diverge near the bottom of the range,
    // where the policy saturates and returns much the same pose for ever-lower
    // commands -- invisible from the console without this.
    //
    // Exact 3-D forward kinematics of each leg, not a sagittal-plane estimate.
    // A planar version was tried first and is WRONG by up to 6 cm in a real
    // squat: the hip_roll and knee link frames carry equal and opposite ~10 deg
    // pitch offsets that cancel only while hip roll and yaw are zero, which is
    // never true in a loaded stance. The table below is transcribed from the
    // compiled g1_23dof model (body_pos, body_quat, joint axis, foot site) and
    // reproduces MuJoCo's own site position to 0.0000 mm over random poses.
    //
    // Joint indices follow deploy.yaml's order: 0-5 left leg, 6-11 right leg,
    // each as (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll) --
    // the same order as the chain, so q[base + i] indexes it directly.
    {
        using Vec3 = Eigen::Vector3f;
        using Quat = Eigen::Quaternionf;
        const auto & q = env->robot->data.joint_pos;

        // Foot site position in the pelvis frame, for one leg.
        auto foot_in_body = [&](int base, float ys) -> Vec3 {
            const Vec3 off[6] = {
                Vec3( 0.00000f,  ys * 0.06445f, -0.10270f),
                Vec3( 0.00000f,  ys * 0.05200f, -0.03046f),
                Vec3( 0.02500f,  0.00000f,      -0.12412f),
                Vec3(-0.07827f,  ys * 0.00215f, -0.17734f),
                Vec3( 0.00000f, -ys * 0.00009f, -0.30001f),
                Vec3( 0.00000f,  0.00000f,      -0.01756f),
            };
            // Link frame orientations. Only hip_roll and knee are non-identity.
            const Quat bq[6] = {
                Quat(1.0f, 0.0f,  0.00000f, 0.0f),
                Quat(0.99618f, 0.0f, -0.08734f, 0.0f),
                Quat(1.0f, 0.0f,  0.00000f, 0.0f),
                Quat(0.99618f, 0.0f,  0.08734f, 0.0f),
                Quat(1.0f, 0.0f,  0.00000f, 0.0f),
                Quat(1.0f, 0.0f,  0.00000f, 0.0f),
            };
            const char axis[6] = { 'y', 'x', 'z', 'y', 'y', 'x' };

            Vec3 p = Vec3::Zero();
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            for (int i = 0; i < 6; ++i) {
                p += R * off[i];
                const Vec3 a = (axis[i] == 'x') ? Vec3::UnitX()
                             : (axis[i] == 'y') ? Vec3::UnitY()
                                                : Vec3::UnitZ();
                R = R * bq[i].toRotationMatrix()
                      * Eigen::AngleAxisf(q[base + i], a).toRotationMatrix();
            }
            return Vec3(p + R * Vec3(0.04f, 0.0f, -0.037f));  // foot site
        };

        // Vertical drop from pelvis to a foot is the dot product of that foot's
        // body-frame position with the unit gravity direction expressed in the
        // body frame, so a leaning robot is handled without extra work.
        // The LOWER foot is the larger drop -- the same reference the training
        // reward uses, so a lifting swing foot does not move the reading.
        const Vec3 & gb = env->robot->data.projected_gravity_b;
        const float achieved = std::max(gb.dot(foot_in_body(0,  1.0f)),
                                        gb.dot(foot_in_body(6, -1.0f)));

        // Debug: command vs achievement, throttled to ~2 Hz (loop is ~50 Hz).
        static int _bh_print = 0;
        if (++_bh_print % 25 == 0) {
            printf("[base_height] cmd = %.3f m | achieved = %.3f m | err = %+.3f m"
                   "  (target %.3f, walking=%s)\n",
                   h, achieved, achieved - h, target,
                   (cmd_target > target + 1e-4f) ? "yes" : "no");
            fflush(stdout);
        }
    }

    return std::vector<float>{ h };
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

}
}