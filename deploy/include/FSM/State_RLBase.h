// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <vector>
#include <chrono>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Upper-body (arm) targets from the IL / GR00T manipulation policy. These are
    // published as a LowCmd on a SEPARATE topic ("rt/arm_targets") so they do not
    // collide with this controller's single rt/lowcmd writer. When the stream is
    // fresh the arm joints follow these targets; otherwise they hold the default
    // (carry) pose. A velocity-limited ramp smooths the carry <-> IL handoff.
    std::shared_ptr<unitree::robot::g1::subscription::LowCmd> arm_sub_;
    std::vector<float> arm_q_applied_;
    bool arm_ramp_init_ = false;
    bool arm_first_run_ = true;
    std::chrono::steady_clock::time_point arm_last_tp_;

    // Remote arm-pose state machine (gamepad Y/B), imported from arm.py:
    //   Y: DEFAULT <-> FORWARD ,  B: FORWARD <-> BOX_HOLD.
    enum class ArmPose { DEFAULT, FORWARD, BOX_HOLD };
    ArmPose arm_pose_ = ArmPose::DEFAULT;
    bool arm_prev_y_ = false;
    bool arm_prev_b_ = false;
};

REGISTER_FSM(State_RLBase)
