"""Bounded two-level plan / execute / replan loop for the Go2 CHAMP stack.

This module implements **Phase 2** of ``docs/LONG_HORIZON_PLANNING_PLAN.md``: a
*plan* is a list of **skills**, each skill is a closed, typed entry that expands
(in trusted code) into the ``MotionCommand`` list the action contract already
produces, and the LLM is re-queried **only** when a skill's **precondition**
fails, under a hard **replan budget**.

Safety invariants (the paper's thesis, preserved exactly)
--------------------------------------------------------
1. **One failure output.**  The refinement chain has exactly one failure
   output: a 1.0 s stationary hold.  The planner preserves that.  Any invalid
   plan, unknown skill, unknown parameter, out-of-range parameter, non-finite
   parameter, budget exhaustion, or failed replan collapses the whole run to
   the stationary hold.  There is no third outcome and no partially-applied
   plan: a plan is validated *in full* before a single command is emitted.
2. **The model never reaches the plant.**  The planner may only *select and
   order* skills from a closed vocabulary.  It never emits code, never emits a
   raw ``MotionCommand``, and never emits a raw number that reaches the
   actuators without passing the parameter validators below.
3. **Preconditions are evaluated in trusted code.**  The LLM may *declare*
   what it expects; the stack decides.  A precondition is a predicate over
   robot state (base position, base attitude, measured velocity, elapsed
   time, faults) -- never a string the model asserts.

Pure logic
----------
Everything in this module except :class:`SkillExecutor` is importable and
unit-testable **without MuJoCo running**: the plan parser, the skill registry,
the parameter validators, the precondition predicates and the replan loop are
pure functions over plain dataclasses.  The MuJoCo-dependent part is confined
to :meth:`SkillExecutor.execute`, which is imported lazily.

The one place this module talks to the simulation is the *state provider*
callable passed into :class:`Planner`; the planner itself never imports
``mujoco``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from go2_champ_control import (
    MotionCommand,
    MotionMode,
    SafetyBounds,
    TrajectorySpec,
    Waypoint,
    wrap_to_pi,
)

__all__ = [
    # Constants
    "MAX_SKILLS_PER_PLAN",
    "DEFAULT_REPLAN_BUDGET",
    "MAX_REPLAN_PROMPT_CHARS",
    "STAND_BY_DURATION_S",
    "STALL_SPEED_THRESHOLD_S",
    "STALL_WINDOW_S",
    "TIMER_TOLERANCE_S",
    "SKILL_NAMES",
    "GAIT_NAMES",
    "TRAJECTORY_PATH_TYPES",
    "PRECONDITION_NAMES",
    "PLAN_SCHEMA_HINT",
    # Data contracts
    "PlanStatus",
    "PreconditionStatus",
    "SkillSpec",
    "Skill",
    "RobotState",
    "PreconditionResult",
    "PlanValidation",
    "PlannerEvent",
    "PlanResult",
    # Registry / validation
    "SKILL_REGISTRY",
    "get_skill_spec",
    "coerce_number",
    "clip_finite",
    "validate_skill_args",
    "parse_plan",
    "build_plan",
    "expand_skill",
    "skill_goal",
    "skill_budget",
    "trajectory_mode_available",
    "gait_modes_available",
    # Preconditions
    "evaluate_precondition",
    # Executor
    "SkillExecutor",
    "SimulatedSkillExecutor",
    "as_stable_waypoint_command",
    "GoalWatchdog",
    # Planner
    "Planner",
    "build_replan_prompt",
    "state_summary",
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: Maximum number of skills accepted in a single plan.  This mirrors the
#: contract's ``MAX_COMMANDS_PER_RESPONSE = 64`` convention: the parse cost is
#: linear in the attacker-controlled payload, so an unbounded plan is a
#: denial-of-service surface on the control path.  A plan longer than this is
#: **rejected** (not truncated) -- truncating would silently drop a subgoal the
#: model asked for, which is exactly the "partially applied plan" the safety
#: argument forbids.
MAX_SKILLS_PER_PLAN = 16

#: Default number of replans allowed per plan (plan section 3.3, ``R = 2``).
DEFAULT_REPLAN_BUDGET = 2

#: Hard cap on the character length of a replan prompt.  The plan's prompt
#: budget must not grow without bound with the plan length, and the replan
#: prompt embeds the plan so far.
MAX_REPLAN_PROMPT_CHARS = 8000

#: Duration of the single failure output: the stationary hold.
STAND_BY_DURATION_S = 1.0


# -----------------------------------------------------------------------------
# Numeric coercion -- the repo's fail-closed idiom
# -----------------------------------------------------------------------------

def coerce_number(value: Any) -> float:
    """Coerce a JSON scalar to a float, rejecting the cases that must fail closed.

    This is the same idiom as ``go2_llm_commands._coerce_number`` and exists
    for the same three reasons:

    * A JSON boolean is not a number.  ``float(True)`` is ``1.0``, so
      ``{"vx": true}`` would otherwise be read as a request for 1.0 m/s and
      then clamped to a legal-looking value -- rather than rejected.
    * An integer literal too large for a float raises ``OverflowError`` inside
      ``float()``.  ``json.loads`` happily produces a 400-digit Python ``int``,
      whose magnitude exceeds the float range.  Left uncaught, that exception
      escapes the validator and the contract is no longer total.
    * A numeric *string* is accepted (providers sometimes quote numbers), but
      its magnitude is still bounded: an out-of-range string saturates to
      ``inf`` and is caught by the caller's finite check.

    Non-finite results are *returned* rather than raised, so the caller's
    single finite check applies one uniform fail-closed policy to every field.

    Raises:
        ValueError: the value is a boolean, or is not numeric at all.
        OverflowError: the value's magnitude exceeds the float range.
    """
    if isinstance(value, bool):
        raise ValueError("a JSON boolean is not a number")
    return float(value)


def clip_finite(value: float, low: float, high: float) -> float:
    """Clip a finite value into ``[low, high]``.

    A *non-finite* value is **not** clamped into range: there is no meaningful
    "nearest legal value" for a NaN, and a NaN that reaches the navigator
    poisons the slew limiter's internal state permanently.  The caller is
    expected to have rejected non-finite values before calling this; this
    function raises if one arrives, so the check cannot be forgotten silently.
    """
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {value!r} cannot be clamped")
    return float(min(max(value, low), high))


# -----------------------------------------------------------------------------
# Skill vocabulary
# -----------------------------------------------------------------------------

class PreconditionStatus(str, Enum):
    """Outcome of evaluating a precondition."""

    MET = "met"
    UNMET = "unmet"
    #: The precondition could not be evaluated because the state it needs was
    #: absent.  Treated exactly like ``UNMET`` by the planner: fail closed.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SkillSpec:
    """The closed definition of one skill.

    A skill is a named, typed entry onto machinery that already exists.  The
    parameter set is closed: an unknown parameter key is a validation failure,
    not something to ignore, because "the model asked for a parameter we do
    not implement" and "the model asked for a parameter we silently dropped"
    are indistinguishable at the plant.
    """

    name: str
    #: Legal parameter names for this skill.  ``None`` means "no parameters".
    params: Tuple[str, ...]
    #: ``parameter name -> (low, high)`` inclusive numeric range.
    ranges: Dict[str, Tuple[float, float]]
    #: Parameters that are required (must be present).
    required: Tuple[str, ...]
    #: Whether the skill maps onto ``MotionMode.TRAJECTORY``.
    needs_trajectory: bool = False
    description: str = ""


#: The closed skill vocabulary.  Every skill expands, in trusted code, into
#: the same ``MotionCommand`` list the action contract already produces.
#:
#: ``follow_trajectory`` and ``gait`` are the two entries that depend on
#: machinery the contract schema may or may not yet expose; both are handled
#: dynamically (see :func:`trajectory_mode_available` and
#: :func:`gait_modes_available`) so the planner does not depend on a
#: particular revision of the contract schema.
SKILL_REGISTRY: Dict[str, SkillSpec] = {
    "stand": SkillSpec(
        name="stand",
        params=("duration",),
        ranges={"duration": (0.1, 60.0)},
        required=("duration",),
        description="Hold station for a bounded duration.",
    ),
    "move_velocity": SkillSpec(
        name="move_velocity",
        params=("vx", "vy", "yaw_rate", "duration"),
        ranges={
            "vx": (-0.30, 0.60),
            "vy": (-0.35, 0.35),
            "yaw_rate": (-0.80, 0.80),
            "duration": (0.1, 60.0),
        },
        required=("vx", "duration"),
        description="Trot at a body-frame velocity for a bounded duration.",
    ),
    "goto_waypoint": SkillSpec(
        name="goto_waypoint",
        params=("x", "y", "tolerance", "speed", "duration"),
        ranges={
            "x": (-10.0, 10.0),
            "y": (-10.0, 10.0),
            "tolerance": (0.05, 0.50),
            "speed": (0.1, 0.60),
            "duration": (0.5, 60.0),
        },
        required=("x", "y", "duration"),
        description="Navigate to a world-frame waypoint under closed-loop control.",
    ),
    "follow_trajectory": SkillSpec(
        name="follow_trajectory",
        params=("path_type", "duration", "radius", "scale_x", "scale_y", "omega", "speed"),
        ranges={
            "duration": (0.5, 60.0),
            "radius": (0.05, 10.0),
            "scale_x": (0.05, 10.0),
            "scale_y": (0.05, 10.0),
            "omega": (0.01, 5.0),
            "speed": (0.05, 0.60),
        },
        required=("path_type", "duration"),
        needs_trajectory=True,
        description="Track a closed parametric path (circle or figure_8).",
    ),
    "gait": SkillSpec(
        name="gait",
        params=("name", "duration"),
        ranges={"duration": (0.1, 60.0)},
        required=("name", "duration"),
        description="Switch the active gait for a bounded duration.",
    ),
}

#: The legal gait names.  Closed tuple, mirroring ``GaitType`` in
#: ``go2_multi_gait`` without importing it (so this module stays importable
#: without numpy-adjacent side effects).
GAIT_NAMES: Tuple[str, ...] = ("trot", "pace", "bound", "pronk", "walk")

#: The legal trajectory path types.  Mirrors ``TRAJECTORY_PATH_TYPES``.
TRAJECTORY_PATH_TYPES: Tuple[str, ...] = ("circle", "figure_8")

SKILL_NAMES: Tuple[str, ...] = tuple(SKILL_REGISTRY.keys())

#: The precondition vocabulary (plan section 3.2).  Six predicates, all
#: evaluated in trusted code over quantities the harness already measures.
PRECONDITION_NAMES: Tuple[str, ...] = (
    "base_height_ok",
    "attitude_ok",
    "arrived",
    "stalled",
    "timer_elapsed",
    "fault_free",
)


def _build_schema_hint() -> str:
    """The plan schema, rendered for the prompt.

    The schema is generated from the registry, so the prompt can never drift
    from the vocabulary the validator enforces.
    """
    lines = [
        '{"explanation": "...", "plan": [',
        '  {"skill": "<name>", "args": {<parameter>: <value>}},',
        '  ...',
        ']}',
        "",
        f"Legal skills (at most {MAX_SKILLS_PER_PLAN} per plan):",
    ]
    for spec in SKILL_REGISTRY.values():
        ranges = ", ".join(
            f"{k} in [{spec.ranges[k][0]:g}, {spec.ranges[k][1]:g}]"
            for k in spec.params
            if k in spec.ranges
        )
        lines.append(
            f"  {spec.name}: args {list(spec.params)}; required {list(spec.required)}"
            + (f"; {ranges}" if ranges else "")
        )
    lines.append(
        f"  Legal path_type values: {list(TRAJECTORY_PATH_TYPES)}; "
        f"legal gait names: {list(GAIT_NAMES)}."
    )
    return "\n".join(lines)


#: The plan schema shown to the model.  Derived from :data:`SKILL_REGISTRY`.
PLAN_SCHEMA_HINT: str = _build_schema_hint()


def get_skill_spec(name: str) -> Optional[SkillSpec]:
    """Return the spec for a skill name, or ``None`` if the name is unknown."""
    return SKILL_REGISTRY.get(name)


def trajectory_mode_available() -> bool:
    """Whether ``MotionMode.TRAJECTORY`` exists in the installed contract.

    The trajectory skill is delegated to the contract's ``trajectory`` mode.
    Another change in this repository is adding that mode concurrently, so
    this module probes for it *dynamically* rather than assuming it exists.
    """
    return hasattr(MotionMode, "TRAJECTORY")


def gait_modes_available() -> bool:
    """Whether the gait field is meaningful on a ``MotionCommand``.

    ``MotionCommand.gait`` already exists in the control module, but the
    contract validator may not yet read it.  The planner probes the contract
    at runtime through this helper so the skill layer does not have to be
    edited when the contract revision changes.
    """
    return "gait" in MotionCommand.__dataclass_fields__


# -----------------------------------------------------------------------------
# Skill
# -----------------------------------------------------------------------------

@dataclass
class Skill:
    """One validated skill invocation.

    ``args`` holds only *validated* values: every numeric parameter is
    finite and inside its declared range.  ``clamped`` records whether any
    parameter was moved by the validator, so the audit trail can tell "the
    model asked for a legal skill" from "the model asked for an illegal one
    and we moved it" -- the same distinction the contract makes.
    """

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    clamped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.name,
            "args": dict(self.args),
            "clamped": self.clamped,
        }

    def describe(self) -> str:
        """One-line human-readable form, used in the replan prompt."""
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.args.items()))
        return f"{self.name}({parts})"


@dataclass
class PlanValidation:
    """Result of validating a plan payload.

    ``ok`` is the only field that decides whether the plan may be executed.
    """

    ok: bool
    skills: List[Skill] = field(default_factory=list)
    error: Optional[str] = None
    clamped: bool = False

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok


# -----------------------------------------------------------------------------
# Robot state and preconditions
# -----------------------------------------------------------------------------

@dataclass
class RobotState:
    """The robot state the preconditions are evaluated over.

    Every field is measured by the harness; none of it comes from the model.
    ``base_pos`` is world-frame (x, y, z) [m]; ``base_euler`` is (roll, pitch,
    yaw) [rad]; ``linvel_body`` is body-frame (vx, vy, vz) [m/s];
    ``elapsed_s`` is the time since the current skill started; ``faults`` is
    the list of safety faults raised during the current skill.
    """

    base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    linvel_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: World-frame goal of the current skill, used by the ``arrived``
    #: precondition.  ``None`` when the skill has no spatial goal.
    goal_xy: Optional[Tuple[float, float]] = None
    #: The goal tolerance [m] for the ``arrived`` precondition.
    goal_tolerance: float = 0.15
    #: Time since the current skill started [s].
    elapsed_s: float = 0.0
    #: The skill's declared time budget [s] (``timer_elapsed``).
    budget_s: float = 0.0
    #: Safety faults raised during the current skill.
    faults: Tuple[str, ...] = ()
    #: Whether the current skill is commanding a non-zero motion (the
    #: ``stalled`` precondition only applies while a motion is commanded).
    commanding_motion: bool = False
    #: How long the measured speed has been below the stall threshold [s].
    stalled_s: float = 0.0
    #: Whether a motion is currently commanded at all (for the summary).
    moving: bool = False


@dataclass
class PreconditionResult:
    """The outcome of one precondition evaluation.

    Polarity
    --------
    Every precondition in the vocabulary has the **same polarity**: ``met`` is
    true when the skill is *proceeding correctly* and the plan may continue.
    The vocabulary contains both "good" predicates (``base_height_ok``,
    ``attitude_ok``, ``arrived``, ``fault_free``) and "alarm" predicates
    (``stalled``, ``timer_elapsed``), and a uniform polarity is what lets the
    planner treat all six uniformly.  For the alarm-flavored two, ``met`` is
    true when the alarm is **not** raised:

    * ``stalled`` is met while the robot is *not* stalled.
    * ``timer_elapsed`` is met while the skill is *still inside* its budget.

    The alarm itself is always reported in ``detail`` and ``value``, so the
    replan prompt can state which condition failed and what it measured.

    Applicability
    -------------
    A precondition that does not apply to the current skill is reported as
    ``applicable=False`` and is **not** a failure.  ``arrived`` needs a
    spatial goal and ``stalled`` needs a commanded motion; a skill with
    neither must not be failed for "not arriving" or "not stalling".
    """

    name: str
    met: bool
    value: Optional[float]
    detail: str = ""
    status: PreconditionStatus = PreconditionStatus.MET
    applicable: bool = True

    @property
    def failed(self) -> bool:
        """True when the precondition applies and is not satisfied."""
        return self.applicable and not self.met

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precondition": self.name,
            "met": self.met,
            "status": self.status.value,
            "applicable": self.applicable,
            "value": self.value,
            "detail": self.detail,
        }


#: Speed below which the robot counts as stalled [m/s].
STALL_SPEED_THRESHOLD_S = 0.05

#: How long the speed must stay below the stall threshold before a stall is
#: declared [s].
STALL_WINDOW_S = 1.0

#: Slack allowed on a skill's time budget before the timer is treated as
#: expired [s].  The executor loop stops *after* the budget elapses, so an
#: exact comparison would fire on every skill that ran to completion.
TIMER_TOLERANCE_S = 0.25


def evaluate_precondition(
    name: str,
    state: RobotState,
    bounds: SafetyBounds,
) -> PreconditionResult:
    """Evaluate one precondition over robot state, in trusted code.

    The six preconditions from the plan's table:

    ``base_height_ok``
        ``bounds.min_height_m <= z <= bounds.max_height_m``.
    ``attitude_ok``
        ``|roll| <= bounds.max_roll_rad`` and ``|pitch| <= bounds.max_pitch_rad``.
    ``arrived``
        ``||p - waypoint|| <= tolerance`` (needs a goal; unavailable without one).
    ``stalled``
        the measured speed has been below the stall threshold for more than
        the stall window while a motion is commanded.
    ``timer_elapsed``
        the time since the skill started exceeds its declared budget.
    ``fault_free``
        no safety fault has been raised during the skill.

    Returns:
        A :class:`PreconditionResult`.  A precondition that cannot be
        evaluated (e.g. ``arrived`` with no goal) reports ``UNAVAILABLE``,
        which the planner treats exactly like ``UNMET``.
    """
    if name == "base_height_ok":
        z = float(state.base_pos[2])
        ok = bounds.min_height_m <= z <= bounds.max_height_m
        return PreconditionResult(
            name=name,
            met=ok,
            value=z,
            detail=f"z={z:.3f} m in [{bounds.min_height_m:.3f}, {bounds.max_height_m:.3f}]",
        )

    if name == "attitude_ok":
        roll = float(state.base_euler[0])
        pitch = float(state.base_euler[1])
        ok = abs(roll) <= bounds.max_roll_rad and abs(pitch) <= bounds.max_pitch_rad
        return PreconditionResult(
            name=name,
            met=ok,
            value=max(abs(roll), abs(pitch)),
            detail=(
                f"roll={roll:.3f} rad (max {bounds.max_roll_rad:.3f}), "
                f"pitch={pitch:.3f} rad (max {bounds.max_pitch_rad:.3f})"
            ),
        )

    if name == "arrived":
        if state.goal_xy is None:
            return PreconditionResult(
                name=name,
                met=True,
                value=None,
                status=PreconditionStatus.UNAVAILABLE,
                applicable=False,
                detail="no spatial goal for this skill",
            )
        gx, gy = state.goal_xy
        px, py = float(state.base_pos[0]), float(state.base_pos[1])
        dist = math.hypot(px - gx, py - gy)
        ok = dist <= float(state.goal_tolerance)
        return PreconditionResult(
            name=name,
            met=ok,
            value=dist,
            detail=(
                f"distance to goal ({gx:.2f}, {gy:.2f}) is {dist:.3f} m "
                f"(tolerance {state.goal_tolerance:.3f} m)"
            ),
        )

    if name == "stalled":
        if not state.commanding_motion:
            return PreconditionResult(
                name=name,
                met=True,
                value=None,
                status=PreconditionStatus.UNAVAILABLE,
                applicable=False,
                detail="no motion commanded, so a stall cannot be declared",
            )
        speed = math.sqrt(
            float(state.linvel_body[0]) ** 2 + float(state.linvel_body[1]) ** 2
        )
        stalled = float(state.stalled_s) >= STALL_WINDOW_S
        return PreconditionResult(
            name=name,
            met=not stalled,
            value=speed,
            detail=(
                f"stalled: measured speed {speed:.3f} m/s stayed below "
                f"{STALL_SPEED_THRESHOLD_S:.3f} m/s for {state.stalled_s:.2f} s"
                if stalled
                else f"measured speed {speed:.3f} m/s, moving"
            ),
        )

    if name == "timer_elapsed":
        elapsed = float(state.elapsed_s)
        budget = float(state.budget_s)
        expired = elapsed > budget + TIMER_TOLERANCE_S
        return PreconditionResult(
            name=name,
            met=not expired,
            value=elapsed,
            detail=(
                f"elapsed {elapsed:.2f} s against a budget of {budget:.2f} s"
                + (" (budget exceeded)" if expired else " (within budget)")
            ),
        )

    if name == "fault_free":
        faults = tuple(state.faults)
        return PreconditionResult(
            name=name,
            met=not faults,
            value=float(len(faults)),
            detail=("no faults" if not faults else f"faults: {list(faults)}"),
        )

    # An unknown precondition name is itself a validation failure upstream;
    # report it as unmet rather than raising, so the planner's fail-closed
    # path handles it uniformly.
    return PreconditionResult(
        name=name,
        met=False,
        value=None,
        status=PreconditionStatus.UNAVAILABLE,
        detail=f"unknown precondition {name!r}",
    )


# -----------------------------------------------------------------------------
# Plan parsing and validation
# -----------------------------------------------------------------------------
# Plan parsing and validation
# -----------------------------------------------------------------------------

def validate_skill_args(
    spec: SkillSpec,
    args: Any,
    bounds: SafetyBounds,
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """Validate one skill's arguments against its closed parameter set.

    The rules, from the plan's section 2.3:

    * **Unknown skill** -> fail closed (handled by the caller).
    * **Unknown parameter** -> fail closed.  An unrecognised key is an
      error, not something to drop: silently dropping it is
      indistinguishable at the plant from obeying it.
    * **Out-of-range parameter** -> *clamped* to the skill's declared
      bounds, and the result is marked as clamped so the audit trail records
      that the value was moved.
    * **Non-finite parameter** -> fail closed, not clamped.  There is no
      meaningful "nearest legal value" for a NaN, and a NaN that reaches
      the navigator poisons the slew limiter's internal state permanently.
    * **Boolean parameter** -> fail closed (``float(True)`` is ``1.0``).

    Returns:
        ``(args, clamped, error)``.  On error, ``args`` is ``None`` and
        ``error`` names the offending field.
    """
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, False, f"skill '{spec.name}' args must be an object, got {type(args).__name__}"

    unknown = [k for k in args if k not in spec.params]
    if unknown:
        return (
            None,
            False,
            f"skill '{spec.name}' has unknown parameter(s) {sorted(unknown)}; "
            f"legal parameters are {list(spec.params)}",
        )

    missing = [k for k in spec.required if k not in args]
    if missing:
        return None, False, f"skill '{spec.name}' is missing required parameter(s) {missing}"

    resolved: Dict[str, Any] = {}
    clamped = False

    for key in spec.params:
        if key not in args:
            continue
        raw = args[key]

        if key == "path_type":
            if not isinstance(raw, str) or raw.strip().lower() not in TRAJECTORY_PATH_TYPES:
                return (
                    None,
                    False,
                    f"skill '{spec.name}' parameter 'path_type' must be one of "
                    f"{list(TRAJECTORY_PATH_TYPES)}, got {raw!r}",
                )
            resolved[key] = raw.strip().lower()
            continue

        if key == "name":
            if not isinstance(raw, str) or raw.strip().lower() not in GAIT_NAMES:
                return (
                    None,
                    False,
                    f"skill '{spec.name}' parameter 'name' must be one of "
                    f"{list(GAIT_NAMES)}, got {raw!r}",
                )
            resolved[key] = raw.strip().lower()
            continue

        try:
            value = coerce_number(raw)
        except (ValueError, TypeError, OverflowError) as exc:
            return None, False, f"skill '{spec.name}' parameter '{key}' is not a number ({exc})"

        if not math.isfinite(value):
            return (
                None,
                False,
                f"skill '{spec.name}' parameter '{key}' is non-finite ({value!r}); "
                "a non-finite value cannot be clamped into range",
            )

        low, high = spec.ranges[key]
        # The skill's declared range is intersected with the contract's
        # SafetyBounds, so a skill can never widen the safety envelope the
        # action contract already enforces.
        if key == "vx":
            low = max(low, bounds.min_vx)
            high = min(high, bounds.max_vx)
        elif key == "vy":
            low = max(low, -bounds.max_vy)
            high = min(high, bounds.max_vy)
        elif key == "yaw_rate":
            low = max(low, -bounds.max_yaw_rate)
            high = min(high, bounds.max_yaw_rate)
        elif key == "duration":
            high = min(high, bounds.max_duration)

        bounded = clip_finite(value, low, high)
        if bounded != value:
            clamped = True
        resolved[key] = bounded

    return resolved, clamped, None


def parse_plan(
    payload: Any,
    bounds: Optional[SafetyBounds] = None,
) -> PlanValidation:
    """Parse and fully validate a plan payload.

    The payload is either a list of skill objects or an object with a
    ``"plan"`` key holding such a list (the wrapper shape from the plan's
    section 2.3).  Each entry is ``{"skill": "<name>", "args": {...}}``.

    The whole plan is validated before anything is returned, so a plan with a
    bad skill in position 7 never produces a partially-applicable prefix.

    Returns:
        A :class:`PlanValidation`.  ``ok`` is ``False`` for an unknown skill,
        an unknown parameter, a non-finite parameter, a missing required
        parameter, an empty plan, or a plan longer than
        :data:`MAX_SKILLS_PER_PLAN`.
    """
    bounds = bounds or SafetyBounds()

    if isinstance(payload, dict):
        if "plan" in payload:
            entries = payload["plan"]
        elif "skills" in payload:
            entries = payload["skills"]
        else:
            entries = None
    else:
        entries = payload

    if not isinstance(entries, list) or not entries:
        return PlanValidation(
            ok=False,
            error=(
                "plan must be a non-empty list of skill objects "
                f"(or an object with a 'plan' key), got {type(entries).__name__}"
            ),
        )

    if len(entries) > MAX_SKILLS_PER_PLAN:
        return PlanValidation(
            ok=False,
            error=(
                f"plan has {len(entries)} skills, exceeding the maximum of "
                f"{MAX_SKILLS_PER_PLAN}"
            ),
        )

    skills: List[Skill] = []
    any_clamped = False

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return PlanValidation(
                ok=False,
                error=f"plan[{idx}] must be an object, got {type(entry).__name__}",
            )

        raw_name = entry.get("skill", entry.get("name"))
        if not isinstance(raw_name, str):
            return PlanValidation(
                ok=False,
                error=f"plan[{idx}] must name a skill with a string, got {raw_name!r}",
            )
        skill_name = raw_name.strip().lower()

        spec = get_skill_spec(skill_name)
        if spec is None:
            return PlanValidation(
                ok=False,
                error=(
                    f"plan[{idx}] names unknown skill {skill_name!r}; "
                    f"legal skills are {list(SKILL_REGISTRY.keys())}"
                ),
            )

        args, clamped, error = validate_skill_args(spec, entry.get("args", {}), bounds)
        if error is not None:
            return PlanValidation(ok=False, error=f"plan[{idx}]: {error}")

        skills.append(Skill(name=spec.name, args=args or {}, clamped=clamped))
        any_clamped = any_clamped or clamped

    return PlanValidation(ok=True, skills=skills, clamped=any_clamped)


def build_plan(skills: Sequence[Skill]) -> Dict[str, Any]:
    """Build the plan wrapper payload from validated skills.

    The inverse of :func:`parse_plan`, used to build the replan prompt and to
    re-parse a plan that came back from the model.
    """
    return {
        "explanation": "",
        "plan": [s.to_dict() for s in skills],
    }


# -----------------------------------------------------------------------------
# Skill expansion into MotionCommands
# -----------------------------------------------------------------------------

def expand_skill(skill: Skill, bounds: Optional[SafetyBounds] = None) -> List[MotionCommand]:
    """Expand one validated skill into the ``MotionCommand`` list it stands for.

    The expansion happens in **trusted** code.  The model's contribution was
    only the *choice* of skill and the values of its declared parameters, all
    of which have already been validated.

    A skill expands to a single command plus the trailing stationary hold
    that the action contract's own invariant requires (a moving sequence must
    end with a stand).  The hold is added by the *plan* level, not here, so
    that the expansion of a single skill is exactly one command.
    """
    bounds = bounds or SafetyBounds()
    args = skill.args
    name = skill.name

    if name == "stand":
        return [MotionCommand(mode=MotionMode.STAND, duration=float(args["duration"]))]

    if name == "move_velocity":
        return [
            MotionCommand(
                mode=MotionMode.VELOCITY,
                vx=float(args["vx"]),
                vy=float(args.get("vy", 0.0)),
                yaw_rate=float(args.get("yaw_rate", 0.0)),
                duration=float(args["duration"]),
            )
        ]

    if name == "goto_waypoint":
        waypoint = Waypoint(
            x=float(args["x"]),
            y=float(args["y"]),
            tolerance=float(args.get("tolerance", 0.15)),
            speed=float(args.get("speed", 0.30)),
        )
        return [
            MotionCommand(
                mode=MotionMode.WAYPOINT,
                waypoint=waypoint,
                duration=float(args["duration"]),
            )
        ]

    if name == "follow_trajectory":
        if not trajectory_mode_available():
            raise ValueError("the contract does not expose a 'trajectory' motion mode")
        path_type = str(args["path_type"])
        params: Dict[str, Any] = {}
        if path_type == "circle":
            if "radius" in args:
                params["radius"] = float(args["radius"])
            if "speed" in args:
                params["speed"] = float(args["speed"])
        else:
            if "scale_x" in args:
                params["scale_x"] = float(args["scale_x"])
            if "scale_y" in args:
                params["scale_y"] = float(args["scale_y"])
            if "omega" in args:
                params["omega"] = float(args["omega"])
        spec = TrajectorySpec(path_type=path_type, params=params, duration=float(args["duration"]))
        return [
            MotionCommand(
                mode=MotionMode.TRAJECTORY,
                trajectory=spec,
                duration=float(args["duration"]),
            )
        ]

    if name == "gait":
        gait_name = str(args["name"])
        return [
            MotionCommand(
                mode=MotionMode.VELOCITY,
                vx=0.0,
                vy=0.0,
                yaw_rate=0.0,
                duration=float(args["duration"]),
                gait=gait_name,
            )
        ]

    raise ValueError(f"unknown skill {name!r}")


def skill_goal(skill: Skill) -> Optional[Tuple[float, float, float]]:
    """Return ``(x, y, tolerance)`` of a skill's spatial goal, or ``None``.

    Used to arm the ``arrived`` precondition.
    """
    if skill.name == "goto_waypoint":
        return (
            float(skill.args["x"]),
            float(skill.args["y"]),
            float(skill.args.get("tolerance", 0.15)),
        )
    return None


def skill_budget(skill: Skill) -> float:
    """The declared time budget of a skill [s].

    For a ``goto_waypoint`` the budget is the skill's own duration (the
    navigator runs closed-loop until it arrives or the window closes).  For
    every other skill it is the commanded duration.
    """
    return float(skill.args.get("duration", 0.0))


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

class PlanStatus(str, Enum):
    """Terminal status of a planning run."""

    #: The plan ran to completion; every skill's preconditions were met.
    SUCCESS = "success"
    #: The run ended in the stationary hold (the single failure output).
    #: ``reason`` says why.
    HELD = "held"


@dataclass
class PlannerEvent:
    """One entry in the planner's event log."""

    kind: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "data": dict(self.data)}


@dataclass
class PlanResult:
    """The outcome of a planning run."""

    status: PlanStatus
    #: The commands that should be executed.  Either the full expansion of an
    #: admissible plan, or exactly the stationary hold.
    commands: List[MotionCommand]
    #: The skills that were executed (expanded), in order.
    executed: List[Skill] = field(default_factory=list)
    #: The number of LLM replans performed.
    replans: int = 0
    #: Whether the run ended in the stationary hold.
    ended_in_hold: bool = False
    #: Why the run ended in the hold (empty if it succeeded).
    hold_reason: str = ""
    #: The event log.
    events: List[PlannerEvent] = field(default_factory=list)
    #: The raw model outputs, in order (initial plan, then each replan).
    raw_outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "ended_in_hold": self.ended_in_hold,
            "hold_reason": self.hold_reason,
            "replans": self.replans,
            "skills_executed": len(self.executed),
            "skills": [s.to_dict() for s in self.executed],
            "commands": [c.to_dict() for c in self.commands],
            "events": [e.to_dict() for e in self.events],
        }


def state_summary(state: RobotState) -> str:
    """A one-paragraph summary of the robot state for the replan prompt.

    Every number here is measured; none of it is model-authored.
    """
    px, py, pz = (float(v) for v in state.base_pos)
    roll, pitch, yaw = (float(v) for v in state.base_euler)
    vx, vy, vz = (float(v) for v in state.linvel_body)
    speed = math.sqrt(vx * vx + vy * vy)
    goal = "none" if state.goal_xy is None else f"({state.goal_xy[0]:.2f}, {state.goal_xy[1]:.2f})"
    faults = list(state.faults)
    return (
        f"position=({px:.2f}, {py:.2f}, {pz:.3f}) m, "
        f"roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f} rad, "
        f"body velocity=({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s (speed {speed:.3f} m/s), "
        f"goal={goal} (tolerance {state.goal_tolerance:.3f} m), "
        f"elapsed={state.elapsed_s:.2f} s of a {state.budget_s:.2f} s budget, "
        f"faults={faults if faults else 'none'}"
    )


def build_replan_prompt(
    original_instruction: str,
    plan_so_far: Sequence[Skill],
    failed_skill: Skill,
    precondition: PreconditionResult,
    state: RobotState,
) -> str:
    """Build the replan prompt.

    Per the plan's section 3.3, the prompt carries:

    * the original instruction,
    * the plan so far,
    * the skill that failed,
    * **which precondition failed and its measured value**, and
    * a robot-state summary.

    The prompt is capped at :data:`MAX_REPLAN_PROMPT_CHARS`.  Truncation
    happens at the *end* (the state summary is last and is the part that
    must survive), and a truncated prompt is a legitimate prompt: it still
    describes the failure, so a model can still answer it.  The cap bounds
    the token cost of a replan, which is the point of a bounded replan.
    """
    executed = ", ".join(s.describe() for s in plan_so_far) or "(nothing executed yet)"

    lines = [
        "The previous plan failed a precondition. Produce a corrected plan.",
        "",
        f"ORIGINAL INSTRUCTION: {original_instruction}",
        "",
        f"PLAN EXECUTED SO FAR: {executed}",
        "",
        f"FAILED SKILL: {failed_skill.describe()}",
        f"FAILED PRECONDITION: {precondition.name}",
        f"MEASURED VALUE: {precondition.detail}",
        "",
        f"ROBOT STATE: {state_summary(state)}",
        "",
        "Reply with a JSON object of the form",
        '{"explanation": "...", "plan": [{"skill": "<name>", "args": {...}}]}',
        "",
        PLAN_SCHEMA_HINT,
        "",
        "Reply with the SHORTEST plan that completes the instruction from the "
        "current state. Do not repeat the skills that already succeeded.",
    ]

    prompt = "\n".join(lines)
    if len(prompt) > MAX_REPLAN_PROMPT_CHARS:
        # Keep the head (instruction, plan, failure) and the tail (state
        # summary, schema): the head says what to fix, the tail says the
        # state and the answer format.
        head_budget = MAX_REPLAN_PROMPT_CHARS // 2
        prompt = (
            prompt[:head_budget]
            + "\n...[truncated]...\n"
            + prompt[-(MAX_REPLAN_PROMPT_CHARS - head_budget - 20):]
        )
    return prompt


class Planner:
    """The two-level plan / execute / replan loop.

    The planner is deliberately *not* coupled to MuJoCo.  It drives an
    executor (a callable that runs one skill and returns the final robot
    state) and an LLM client (a callable that turns a prompt into raw model
    text).  Both are injected, which is what makes the whole loop unit-
    testable without a simulator.

    The loop:

    1. Ask the model for a plan (raw text).
    2. Parse and validate the plan **in full**.  A plan that does not
       validate collapses the run to the stationary hold.
    3. Execute the skills in order.  After each skill, evaluate its
       preconditions.  If a precondition fails, replan (bounded).
    4. If the replan budget is exhausted, or the replan output does not
       validate, abort the whole run to the stationary hold.

    The LLM is re-queried **only** when a skill's precondition fails.
    """

    def __init__(
        self,
        query: Callable[[str], str],
        execute: Callable[[Skill], RobotState],
        bounds: Optional[SafetyBounds] = None,
        replan_budget: int = DEFAULT_REPLAN_BUDGET,
        preconditions: Sequence[str] = PRECONDITION_NAMES,
        logger: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            query: called with a prompt, returns the raw model text.
            execute: called with one skill, runs it and returns the final
                robot state observed at the end of the skill.
            bounds: the safety bounds used for clamping and for the
                ``base_height_ok`` / ``attitude_ok`` preconditions.
            replan_budget: the maximum number of replans per plan.
            preconditions: the preconditions checked after every skill.
                ``arrived`` / ``stalled`` / ``timer_elapsed`` are checked
                too when the skill declares the state they need.
            logger: optional sink for human-readable progress lines.
        """
        self.query = query
        self.execute = execute
        self.bounds = bounds or SafetyBounds()
        self.replan_budget = max(0, int(replan_budget))
        self.preconditions = tuple(preconditions)
        self.logger = logger

    # -- helpers ------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    @staticmethod
    def _hold(bounds: Optional[SafetyBounds] = None) -> List[MotionCommand]:
        """The single failure output: a 1.0 s stationary hold."""
        return [MotionCommand(mode=MotionMode.STAND, duration=STAND_BY_DURATION_S)]

    def _extract_json(self, raw: str) -> Any:
        """Pull the plan object out of a raw model response.

        The model is asked for a JSON object, but it may wrap it in prose or
        a code fence.  Only the *structure* is extracted here; every value is
        validated by :func:`parse_plan` afterwards.  A response with no
        extractable JSON fails closed.
        """
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        # Strip a fenced code block if present.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    # Fall through: try the next bracket pair.
                    continue
        return None

    def _armed_state(self, skill: Skill, state: RobotState) -> RobotState:
        """Arm the skill's goal and budget onto the state.

        The ``arrived`` precondition needs to know what "arrived" means for
        *this* skill, and the ``timer_elapsed`` precondition needs this
        skill's declared budget.  Both come from the skill itself, not from
        the model, so arming them here is trusted bookkeeping.
        """
        goal = skill_goal(skill)
        budget = skill_budget(skill)
        return replace(
            state,
            goal_xy=None if goal is None else (goal[0], goal[1]),
            goal_tolerance=0.15 if goal is None else goal[2],
            budget_s=budget,
        )

    def _check_preconditions(
        self,
        skill: Skill,
        state: RobotState,
    ) -> List[PreconditionResult]:
        """Evaluate every precondition for a skill."""
        armed = self._armed_state(skill, state)
        results: List[PreconditionResult] = []
        for name in self.preconditions:
            results.append(evaluate_precondition(name, armed, self.bounds))
        return results

    # -- the loop -----------------------------------------------------------

    def run(self, instruction: str) -> PlanResult:
        """Run the plan / execute / replan loop for one instruction."""
        events: List[PlannerEvent] = []
        raw_outputs: List[str] = []

        # --- 1. initial plan ------------------------------------------------
        raw = self.query(instruction)
        raw_outputs.append(raw)
        validation = parse_plan(self._extract_json(raw), self.bounds)

        if not validation.ok:
            reason = f"initial plan rejected: {validation.error}"
            events.append(PlannerEvent(kind="plan_rejected", detail=reason))
            self._log(reason)
            return PlanResult(
                status=PlanStatus.HELD,
                commands=self._hold(self.bounds),
                ended_in_hold=True,
                hold_reason=reason,
                events=events,
                raw_outputs=raw_outputs,
            )

        events.append(
            PlannerEvent(
                kind="plan_accepted",
                detail=f"{len(validation.skills)} skill(s)",
                data={"clamped": validation.clamped, "skills": [s.to_dict() for s in validation.skills]},
            )
        )

        executed: List[Skill] = []
        pending: List[Skill] = list(validation.skills)
        replans = 0
        last_state = RobotState()

        while True:
            if not pending:
                break

            skill = pending.pop(0)
            try:
                state = self.execute(skill)
            except Exception as exc:  # pragma: no cover - executor contract
                reason = f"skill {skill.name!r} raised {type(exc).__name__}: {exc}"
                events.append(PlannerEvent(kind="execution_error", detail=reason))
                self._log(reason)
                return PlanResult(
                    status=PlanStatus.HELD,
                    commands=self._hold(self.bounds),
                    executed=executed,
                    replans=replans,
                    ended_in_hold=True,
                    hold_reason=reason,
                    events=events,
                    raw_outputs=raw_outputs,
                )
            last_state = state
            executed.append(skill)

            failures = [r for r in self._check_preconditions(skill, state) if r.failed]
            if not failures:
                events.append(
                    PlannerEvent(kind="skill_ok", detail=skill.describe(), data={"skill": skill.to_dict()})
                )
                continue

            failed = failures[0]
            events.append(
                PlannerEvent(
                    kind="precondition_failed",
                    detail=f"{failed.name}: {failed.detail}",
                    data={"skill": skill.to_dict(), "precondition": failed.to_dict()},
                )
            )
            self._log(f"precondition failed: {failed.name} ({failed.detail})")

            # --- 2. replan (bounded) ----------------------------------------
            if replans >= self.replan_budget:
                reason = (
                    f"replan budget exhausted after {replans} replan(s); "
                    f"precondition {failed.name!r} failed: {failed.detail}"
                )
                events.append(PlannerEvent(kind="hold", detail=reason))
                self._log(reason)
                return PlanResult(
                    status=PlanStatus.HELD,
                    commands=self._hold(self.bounds),
                    executed=executed,
                    replans=replans,
                    ended_in_hold=True,
                    hold_reason=reason,
                    events=events,
                    raw_outputs=raw_outputs,
                )

            prompt = build_replan_prompt(
                instruction, executed, skill, failed, self._armed_state(skill, state)
            )
            raw = self.query(prompt)
            raw_outputs.append(raw)
            replans += 1

            replan = parse_plan(self._extract_json(raw), self.bounds)
            if not replan.ok:
                reason = f"replan rejected: {replan.error}"
                events.append(PlannerEvent(kind="replan_rejected", detail=reason))
                self._log(reason)
                return PlanResult(
                    status=PlanStatus.HELD,
                    commands=self._hold(self.bounds),
                    executed=executed,
                    replans=replans,
                    ended_in_hold=True,
                    hold_reason=reason,
                    events=events,
                    raw_outputs=raw_outputs,
                )

            events.append(
                PlannerEvent(
                    kind="replan_accepted",
                    detail=f"{len(replan.skills)} skill(s)",
                    data={"clamped": replan.clamped, "skills": [s.to_dict() for s in replan.skills]},
                )
            )

            # The remaining plan is replaced wholesale by the replan.  The
            # replan is expected to describe the task *from the current
            # state*; splicing the old tail onto the new plan would be a
            # partially-applied plan, which the safety argument forbids.
            pending = list(replan.skills)

        # --- 3. success -----------------------------------------------------
        commands = self._expand(executed)
        events.append(
            PlannerEvent(
                kind="plan_complete",
                detail=f"{len(executed)} skill(s) executed",
            )
        )
        return PlanResult(
            status=PlanStatus.SUCCESS,
            commands=commands,
            executed=executed,
            replans=replans,
            ended_in_hold=False,
            events=events,
            raw_outputs=raw_outputs,
        )

    def _expand(self, skills: Sequence[Skill]) -> List[MotionCommand]:
        """Expand a fully executed plan into the command list, with the trailing hold.

        The trailing 1.0 s stationary hold is the contract's own invariant
        (a moving sequence must end with a stand), and it is also the shape
        the failure output takes, so a successful run and a failed run have
        the same terminal command.
        """
        commands: List[MotionCommand] = []
        for skill in skills:
            commands.extend(expand_skill(skill, self.bounds))
        if not commands or commands[-1].mode != MotionMode.STAND:
            commands.append(MotionCommand(mode=MotionMode.STAND, duration=STAND_BY_DURATION_S))
        return commands


# -----------------------------------------------------------------------------
# Executors
# -----------------------------------------------------------------------------

class SkillExecutor:
    """Interface for something that can run a skill.

    The planner talks to this through a plain callable, so the concrete
    executor can be a MuJoCo simulation, a stub, or a recorded trace.
    """

    def __call__(self, skill: Skill) -> RobotState:  # pragma: no cover - interface
        raise NotImplementedError


def as_stable_waypoint_command(command: MotionCommand) -> MotionCommand:
    """Return ``command`` in the form the controller's navigator keeps.

    The controller compares waypoint *lists* before re-arming its navigator,
    so a command carrying a stable list keeps the navigator's state across
    ticks.  The single-waypoint form re-arms the navigator on every tick,
    which resets the arrival test and makes the robot orbit its goal instead
    of stopping at it.

    Non-waypoint commands are returned unchanged.
    """
    if command.mode != MotionMode.WAYPOINT or command.waypoint is None:
        return command
    return MotionCommand(
        mode=MotionMode.WAYPOINT,
        waypoints=[command.waypoint],
        waypoint=command.waypoint,
        duration=command.duration,
    )


class GoalWatchdog:
    """Track whether a skill has reached its goal, and disarm the stall timer.

    The plan document defines the ``stalled`` precondition as "the measured
    speed is below epsilon for more than T seconds *while a motion is
    commanded*".  A ``goto_waypoint`` skill that has arrived is no longer
    commanding a motion, so idling at the goal is not a stall.  Read
    literally, the plan would make every waypoint skill that arrives before
    its window closes report a spurious stall, and every waypoint plan would
    replan.

    The latch is **sticky**: once the goal has been reached, the skill
    commands no further motion for the rest of its window, however the
    underlying command object is phrased.
    """

    def __init__(self, goal: Optional[Tuple[float, float, float]] = None):
        #: ``(x, y, tolerance)`` of the goal, or ``None`` for a skill with no
        #: spatial goal.
        self.goal = goal
        #: Whether the goal has been reached.
        self.reached = False

    def update(self, base_pos: Sequence[float]) -> bool:
        """Update the latch from the measured base position.

        Returns whether the goal is reached.  A skill with no goal is never
        "reached" and never disarms the stall watchdog.
        """
        if self.goal is None:
            return False
        if self.reached:
            return True
        gx, gy, tol = self.goal
        self.reached = math.hypot(
            float(base_pos[0]) - gx, float(base_pos[1]) - gy
        ) <= tol
        return self.reached


class SimulatedSkillExecutor:
    """Run a skill in a MuJoCo simulation via ``Go2ChampController``.

    This class is the only MuJoCo-dependent part of the module, and it is
    imported lazily so the rest of the module stays importable without a
    simulator.  It is also not used by the unit tests or by the offline
    experiment, both of which drive the planner through a stub executor.
    """

    def __init__(self, runner: Any, dt: float = 0.002, decimation: int = 1):
        self.runner = runner
        self.dt = dt
        self.decimation = max(1, int(decimation))

    def __call__(self, skill: Skill) -> RobotState:
        import numpy as np  # local import: keeps the module import-light
        import mujoco
        from go2_champ_control import quaternion_to_euler_xyz

        runner = self.runner
        commands = expand_skill(skill)
        # Expand to a single command of the skill's full duration: the
        # waypoint navigator and the trajectory tracker both run closed-loop
        # inside the controller, so the skill's duration is the wall-clock
        # window for the whole skill.
        goal = skill_goal(skill)
        budget = skill_budget(skill)

        # Hand the waypoint command to the controller in its *list* form, so
        # the navigator keeps its state across ticks.
        commands = [as_stable_waypoint_command(commands[0])]

        start_time = float(runner.data.time)
        last_state = RobotState()
        stall_timer = 0.0
        faults: List[str] = []
        watchdog = GoalWatchdog(goal)

        while (float(runner.data.time) - start_time) < budget:
            base_pos = runner.data.qpos[:3].copy()
            base_quat = runner.data.qpos[3:7].copy()
            base_euler = quaternion_to_euler_xyz(base_quat)
            base_rot = runner.data.xmat[runner.base_body_id].reshape(3, 3).copy()
            linvel_body = base_rot.T @ runner.data.qvel[:3].copy()
            angvel_body = runner.data.qvel[3:6].copy()

            joint_pos = np.array([runner.data.qpos[a] for a in runner.qpos_addrs])

            target_q, _desired_feet, _st, _sw, info = runner.controller.step(
                command=commands[0],
                step_time=float(runner.data.time),
                dt=self.dt,
                base_pos=base_pos,
                base_euler=base_euler,
                base_linvel_body=linvel_body,
                base_angvel_body=angvel_body,
                joint_positions=joint_pos,
            )
            runner.data.ctrl[runner.actuator_ids] = target_q

            mujoco.mj_step(runner.model, runner.data)

            fault = info.get("fault")
            if fault:
                faults.append(str(fault))

            speed = float(np.sqrt(linvel_body[0] ** 2 + linvel_body[1] ** 2))
            commanding = bool(
                commands[0].mode != MotionMode.STAND
                and (
                    abs(commands[0].vx) > 1e-6
                    or abs(commands[0].vy) > 1e-6
                    or abs(commands[0].yaw_rate) > 1e-6
                    or commands[0].mode == MotionMode.WAYPOINT
                    or commands[0].mode == MotionMode.TRAJECTORY
                )
            )

            # The stall watchdog is disarmed once the skill has reached its
            # goal: a skill that has arrived is no longer commanding a
            # motion, so idling at the goal is not a stall.  See
            # :class:`GoalWatchdog` for why the plan's literal reading is
            # disambiguated this way.
            goal_reached = watchdog.update(base_pos)
            if goal_reached:
                commanding = False

            if commanding and speed < STALL_SPEED_THRESHOLD_S:
                stall_timer += self.dt
            else:
                stall_timer = 0.0

            # A `goto_waypoint` skill ends the moment its goal is reached, the
            # same way the benchmark harness ends a plan skill: the plan is a
            # sequence of subgoals, so idling at a goal for the remainder of
            # the window would waste the budget of the next skill.  The
            # controller reports STAND once the navigator is done, which is
            # the independent confirmation that the goal is complete.
            skill_complete = goal_reached and str(info.get("mode")) == "stand"

            last_state = RobotState(
                base_pos=(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])),
                base_euler=(float(base_euler[0]), float(base_euler[1]), float(base_euler[2])),
                linvel_body=(
                    float(linvel_body[0]),
                    float(linvel_body[1]),
                    float(linvel_body[2]),
                ),
                goal_xy=None if goal is None else (goal[0], goal[1]),
                goal_tolerance=0.15 if goal is None else goal[2],
                elapsed_s=float(runner.data.time) - start_time,
                budget_s=budget,
                faults=tuple(faults),
                commanding_motion=commanding,
                stalled_s=stall_timer,
                moving=commanding,
            )

            if skill_complete:
                break

        return last_state
