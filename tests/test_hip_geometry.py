"""Regression test for the hip-offset correction.

The constant ``HIP_POSITIONS`` is the position of the hip ROLL axis in the
body frame.  It previously held the roll-to-pitch distance instead, which
made the reported foot positions laterally wrong by about 48 mm and made the
Raibert lever arm 2.05x too long.

The test pins the constant to the model it is derived from, rather than to a
number typed by hand, so the two cannot drift apart again.  The comparison
uses the model's *local* body positions, not the settled world positions, so
it is independent of where the base happens to come to rest.
"""

import mujoco
import pytest

from go2_champ_control import (
    FootholdPlanner,
    GaitConfig,
    HIP_OFFSET,
    HIP_POSITIONS,
    LEG_NAMES,
)

MODEL_PATH = "unitree_go2/go2_description.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def _body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"{name} not found in the model"
    return bid


def test_hip_positions_match_the_models_roll_axis(model):
    """HIP_POSITIONS must be the roll axis, as its own comment says."""
    for i, name in enumerate(LEG_NAMES):
        hip = _body_id(model, f"{name}_hip")
        # body_pos is the body's origin in its parent's frame; the hip bodies
        # are children of the base, so this is the base-frame position.
        local = model.body_pos[hip]
        assert HIP_POSITIONS[i][0] == pytest.approx(float(local[0]), abs=1e-6), (
            f"{name}: HIP_POSITIONS fore-aft coordinate is {HIP_POSITIONS[i][0]:+.4f} "
            f"but the model's roll axis is at {float(local[0]):+.4f}"
        )
        assert HIP_POSITIONS[i][1] == pytest.approx(float(local[1]), abs=1e-6), (
            f"{name}: HIP_POSITIONS lateral coordinate is {HIP_POSITIONS[i][1]:+.4f} "
            f"but the model's roll axis is at {float(local[1]):+.4f}"
        )


def test_hip_offset_is_the_roll_to_pitch_distance(model):
    """HIP_OFFSET must be the roll-to-pitch distance, and must not be a
    duplicate of the roll axis position.

    Conflating the two is the defect this test guards against: they used to
    hold the same number.
    """
    for i, name in enumerate(LEG_NAMES):
        hip = _body_id(model, f"{name}_hip")
        thigh = _body_id(model, f"{name}_thigh")
        # The thigh is a child of the hip, so its own local lateral offset IS
        # the roll-to-pitch distance.
        distance = abs(float(model.body_pos[thigh][1]))
        assert HIP_OFFSET == pytest.approx(distance, abs=1e-6)
        # And the pitch axis therefore sits at the roll axis plus that
        # distance, which is where the feet are.
        pitch_axis_y = abs(float(model.body_pos[hip][1])) + distance
        assert pitch_axis_y == pytest.approx(0.142, abs=1e-6)
        assert abs(HIP_OFFSET - abs(HIP_POSITIONS[i][1])) > 1e-3


def test_reported_foot_positions_match_the_model(model):
    """The foot positions the planner reports must be the model's feet."""
    planner = FootholdPlanner(GaitConfig())
    _, desired_feet, _, _ = planner.plan_leg_footholds(
        cmd_vx=0.0,
        cmd_vy=0.0,
        cmd_yaw_rate=0.0,
        meas_vx=0.0,
        meas_vy=0.0,
        meas_yaw_rate=0.0,
        stance_phases=[0.5] * 4,
        swing_phases=[0.0] * 4,
        stride_period=0.44,
    )
    assert len(desired_feet) == 4
    for i, name in enumerate(LEG_NAMES):
        hip = _body_id(model, f"{name}_hip")
        thigh = _body_id(model, f"{name}_thigh")
        foot = _body_id(model, f"{name}_foot")
        # The foot is directly below the thigh pitch axis, so its lateral
        # coordinate is the hip roll axis plus the roll-to-pitch distance.
        expected_y = float(model.body_pos[hip][1]) + float(model.body_pos[thigh][1])
        reported_y = float(desired_feet[i][1])
        assert reported_y == pytest.approx(expected_y, abs=1e-6), (
            f"{name}: reported foot lateral position {reported_y:+.4f} vs the "
            f"model's {expected_y:+.4f}"
        )
