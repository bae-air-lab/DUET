"""keyboard_to_wireless.py — keyboard → rt/wirelesscontroller DDS publisher.

Keys:
  1           : F1 pulse → FixStand.  From Passive this arms the robot; from
                Velocity it is the HARNESS HOLD: the legs ease into a fixed
                stance over 3 s and hold it under PD. Press this BEFORE lifting
                the robot -- with the feet off the floor the policy has no
                ground reaction and will thrash. Press 2 to hand back afterwards.
  2           : F2 pulse (FixStand → Velocity, policy active)
  Enter       : start pulse (any state → Passive, soft e-stop; legs go LIMP)
  W / S       : vx forward / backward       (ly)
  A / D       : vy left / right             (lx)
  Q / E       : wz yaw left / right         (rx)
  Space       : zero stick velocities
  Up / Down   : D-pad up/down  → raise / lower height target (hold to ramp)
  Left        : D-pad left     → walk-height preset
  Right       : D-pad right    → stand preset
  Y           : Y button → arm toggle DEFAULT <-> FORWARD (reach out for box)
  B           : B button → arm toggle FORWARD <-> BOX_HOLD (grip / loosen)
  X           : X button pulse
  Ctrl-C      : quit
"""
import time, threading, sys, termios, tty, select
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

# NETWORK_IFACE = "enp132s0"
NETWORK_IFACE = "lo"
PUBLISH_HZ = 100

VX_MAX = 0.4
VY_MAX = 0.3
WZ_MAX = 0.6
# Height presets, in metres of pelvis height above the feet. Standing and
# walking (forward, backward, strafe and turn) both use RY_STAND, so the robot
# keeps one posture whether it is holding station or moving.
#
# READ THIS BEFORE CHANGING THEM. These constants are currently COSMETIC: they
# are carried on the `ry` axis of the WirelessController message, and the
# deployed controller never reads `ry`. base_height_command in
# deploy/include/isaaclab/envs/mdp/observations/observations.h builds the
# height from D-pad press events and its own integrator, then clamps the result
# to `commands.base_height.range` in deploy.yaml. So the values that actually
# reach the policy are:
#   idle standing  -> deploy.yaml `commands.base_height.standstill`  (0.78)
#   D-pad Up/Down  -> integrated, then CLAMPED to `range`              (0.24..0.78)
#   D-pad Right    -> `range` upper bound                              (0.78)
#   D-pad Left     -> the literal in observations.h                    (0.60)
#   walking floor  -> WALK_MIN in observations.h, clamps the target UP (0.60)
# Walking inherits the standing target, so with standstill at 0.78 the robot
# walks at 0.78 unless you press Left. The trained-range clamp lives in
# deploy.yaml; these constants mirror it so the two cannot silently disagree.
# Changing the numbers here keeps this file self-consistent and documents the
# intended posture, but it does not move the robot on its own.
# Every value here is inside the TRAINED band (0.24 .. 0.78) of the CURRENTLY
# deployed policy (rough_seed1_50k_35000). Move these only when deploy.yaml moves. Nothing above
# 0.78 and nothing below 0.24: outside that the policy extrapolates, and a deep
# extrapolated squat drives the knee and hip past their soft limits, which on
# hardware becomes a held stall current rather than a visible failure.
RY_STAND = 0.78   # default standing AND walking height; top of the trained band
RY_SQUAT = 0.24   # deepest TRAINED squat; deploy.yaml clamps the D-pad here
RY_MID   = 0.60   # D-pad Left preset, the walk-band floor (set in observations.h)

# Bit positions — verified uint16-safe
KEY_START = 1 << 2   # bit 2
KEY_F1    = 1 << 6   # bit 6
KEY_F2    = 1 << 7   # bit 7
KEY_Y     = 1 << 11   # bit 11 — Y button
KEY_X     = 1 << 10   # bit 10 — X button
KEY_B     = 1 << 9    # bit 9  — B button
KEY_A     = 1 << 8    # bit 8  — A button
# D-pad bits (BtnUnion layout: up=12, right=13, down=14, left=15).
# These drive base_height_command in the C++ deploy (observations.h):
#   hold Up/Down = raise/lower height target, Left = squat preset, Right = stand.
KEY_UP    = 1 << 12
KEY_RIGHT = 1 << 13
KEY_DOWN  = 1 << 14
KEY_LEFT  = 1 << 15
ARROW_HOLD_S = 0.25  # keep the bit set this long after each (auto-repeated) press

PULSE_FRAMES = 15

state = {
    "lx": 0.0, "ly": 0.0, "rx": 0.0,
    "ry": RY_STAND,
    "pulse_bits": 0,
    "pulse_remaining": 0,
    "arrow_bits": 0,
    "arrow_until": 0.0,
    "running": True,
}
lock = threading.Lock()

def hold_arrow(bits, label):
    """Terminal auto-repeat refreshes this while the key is held, so the bit
    stays set for held keys and decays ARROW_HOLD_S after release."""
    with lock:
        state["arrow_bits"] = bits
        state["arrow_until"] = time.time() + ARROW_HOLD_S
    print(f"  [{label}]")

def fire_pulse(bits, label):
    with lock:
        state["pulse_bits"] |= bits      # OR with existing bits
        state["pulse_remaining"] = max(state["pulse_remaining"], PULSE_FRAMES)
    print(f"  [{label}]")

def keyboard_loop():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("Keyboard publisher running.")
        print("  1=F1(FixStand)  2=F2(Velocity)  Enter=Passive")
        print("  WASD=move  Q/E=yaw  Space=stop")
        print(f"  Arrows: Up/Down=height  Left=walk({RY_MID:.2f})  "
              f"Right=stand({RY_STAND:.2f})  [effective values live in deploy.yaml]")
        print("  Y=arm Default/Forward  B=arm Forward/BoxHold  X=X btn")
        print("  Ctrl-C=quit\n")
        while state["running"]:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)

            # Arrow keys arrive as ESC [ A/B/C/D — map to D-pad button bits.
            if ch == '\x1b':
                seq = sys.stdin.read(2)
                if seq == '[A':
                    hold_arrow(KEY_UP, "UP arrow → height +")
                elif seq == '[B':
                    hold_arrow(KEY_DOWN, "DOWN arrow → height -")
                elif seq == '[C':
                    hold_arrow(KEY_RIGHT, "RIGHT arrow → STAND preset")
                elif seq == '[D':
                    hold_arrow(KEY_LEFT, f"LEFT arrow → WALK preset ({RY_MID:.2f})")
                continue

            low = ch.lower()

            # Velocity / height
            if   low == 'w':
                state["ly"] = VX_MAX;   print(f"  [W] ly={VX_MAX:+.2f}")
            elif low == 's':
                state["ly"] = -VX_MAX;  print(f"  [S] ly={-VX_MAX:+.2f}")
            elif low == 'a':
                state["lx"] = -VY_MAX;  print(f"  [A] lx={-VY_MAX:+.2f}")
            elif low == 'd':
                state["lx"] = VY_MAX;   print(f"  [D] lx={VY_MAX:+.2f}")
            elif low == 'q':
                state["rx"] = -WZ_MAX;  print(f"  [Q] rx={-WZ_MAX:+.2f}")
            elif low == 'e':
                state["rx"] = WZ_MAX;   print(f"  [E] rx={WZ_MAX:+.2f}")
            elif low == ' ':
                with lock:
                    state["lx"] = state["ly"] = state["rx"] = 0.0
                print("  [SPACE] zero velocity")
            # Gamepad face buttons — sent as real WirelessController key bits,
            # exactly what g1_ctrl reads (Y=bit11, B=bit9, X=bit10).
            elif low == 'y':
                fire_pulse(KEY_Y, "Y → arm toggle DEFAULT/FORWARD")
            elif low == 'b':
                fire_pulse(KEY_B, "B → arm toggle FORWARD/BOX_HOLD")
            elif low == 'x':
                fire_pulse(KEY_X, "X → X button")


            # FSM transitions
            elif ch == '1':
                fire_pulse(KEY_F1, "1 → F1 (FixStand)")
            elif ch == '2':
                fire_pulse(KEY_F2, "2 → F2 (Velocity)")
            elif ch in ('\r', '\n'):
                fire_pulse(KEY_START, "Enter → Passive")

            elif ch == '\x03':
                state["running"] = False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main():
    ChannelFactoryInitialize(0, NETWORK_IFACE)
    pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    pub.Init()

    t = threading.Thread(target=keyboard_loop, daemon=True)
    t.start()

    period = 1.0 / PUBLISH_HZ
    try:
        while state["running"]:
            msg = unitree_go_msg_dds__WirelessController_()
            with lock:
                msg.lx = state["lx"]
                msg.ly = state["ly"]
                msg.rx = state["rx"]
                msg.ry = state["ry"]
                arrows = state["arrow_bits"] if time.time() < state["arrow_until"] else 0
                msg.keys = state["pulse_bits"] | arrows
                if state["pulse_remaining"] > 0:
                    state["pulse_remaining"] -= 1
                else:
                    state["pulse_bits"] = 0
            pub.Write(msg)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    state["running"] = False
    time.sleep(0.2)
    print("\nStopped.")

if __name__ == "__main__":
    main()


