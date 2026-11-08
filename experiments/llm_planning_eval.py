#!/usr/bin/env python3
"""Offline evaluation of the bounded two-level plan / execute / replan loop.

Question
--------
Does the plan / execute / replan loop *work*, and does it keep the paper's
central guarantee when it fails?

The script measures, over a corpus of multi-step tasks:

* whether the task succeeded,
* how many skills were executed,
* how many replans the loop needed, and
* whether the run ended in the stationary hold -- the contract's **single
  failure output**.

Design
------
**No live LLM.**  The corpus is driven by an offline mock client that returns
canned, deterministic model responses, so the measurement is of the *planner*,
not of the model.  This is the same division of labour the repo's existing
experiment scripts use: `experiments/adversarial_llm_eval.py` sweeps the
parser, and `experiments/language_ablation.py` sweeps the interface.

**A simulated plant.**  The executor is a *deterministic kinematic plant*
rather than MuJoCo: a first-order velocity-tracking model that moves the
robot toward a skill's goal at a bounded speed, so a task either reaches its
subgoal or does not, reproducibly and without the cost of a physics run.  The
point of this experiment is the *control flow* of the loop -- does it replan,
does it recover, does it fail closed -- not the biomechanics of the gait
stack.  (The MuJoCo-backed executor lives in `go2_llm_planner.SimulatedSkillExecutor`
and is not used here.)

**A corpus with a designed failure profile.**  Each task declares the skills
the model returns, and optionally a *fault injection* that makes a particular
skill fail its precondition.  The corpus therefore covers, deliberately:

* clean multi-step tasks that need no replan,
* tasks whose first plan fails a precondition and whose replan recovers,
* tasks whose replan budget is exhausted, and
* tasks whose plan is malformed, so the loop must fail closed.

Usage
-----
    python experiments/llm_planning_eval.py
    python experiments/llm_planning_eval.py --out report/figures/planning_results.json

Exit code is 0 when every task behaved as its class requires, and 1 when any
task's outcome contradicts its declared class (so this doubles as a CI
check).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bootstrap() -> str:
    """Resolve the repo root and put import roots on sys.path.

    Same bootstrap as `experiments/language_ablation.py`.
    """
    root = REPO_ROOT if os.path.exists(os.path.join(REPO_ROOT, "unitree_go2")) else os.getcwd()
    for p in (str(root), os.path.join(str(root), "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


_ROOT = _bootstrap()

from go2_champ_control import MotionMode, SafetyBounds  # noqa: E402
from go2_llm_planner import (  # noqa: E402
    DEFAULT_REPLAN_BUDGET,
    MAX_SKILLS_PER_PLAN,
    PLAN_SCHEMA_HINT,
    PRECONDITION_NAMES,
    SKILL_REGISTRY,
    PlanStatus,
    Planner,
    RobotState,
    build_replan_prompt,
    parse_plan,
    state_summary,
)


# ---------------------------------------------------------------------------
# The plant
# ---------------------------------------------------------------------------

#: How fast the simulated plant tracks a commanded velocity [1/s].  A
#: first-order lag: the measured velocity approaches the command with this
#: time constant, so a stall is a genuine failure of the plant rather than a
#: scripted event.
PLANT_TRACKING_GAIN = 2.0

#: The plant's speed ceiling [m/s].
PLANT_MAX_SPEED = 0.45

#: The plant's maximum waypoint capture radius [m].  The effective capture
#: radius for a skill is ``min(this, the skill's own tolerance)``, so the
#: plant always stops *within* the tolerance the skill declared.
PLANT_CAPTURE_RADIUS = 0.12

#: The plant's integration step [s].
PLANT_DT = 0.05

#: The characters-per-token rule used to *estimate* the prompt and response
#: token counts.  The script runs with no tokenizer available, and the model
#: that will actually serve the request is not the one this script mocks, so
#: a character count divided by this rule is the honest measure available
#: offline.  It is reported as an estimate, never as a tokenizer count.
CHARS_PER_TOKEN = 4.0


@dataclass
class PlantState:
    """Mutable state of the simulated plant."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.28
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    faults: List[str] = field(default_factory=list)

    def to_robot_state(
        self,
        elapsed_s: float,
        budget_s: float,
        commanding_motion: bool,
        stalled_s: float,
    ) -> RobotState:
        return RobotState(
            base_pos=(self.x, self.y, self.z),
            base_euler=(self.roll, self.pitch, self.yaw),
            linvel_body=(self.vx, self.vy, 0.0),
            elapsed_s=elapsed_s,
            budget_s=budget_s,
            faults=tuple(self.faults),
            commanding_motion=commanding_motion,
            stalled_s=stalled_s,
            moving=commanding_motion,
        )


class SimulatedPlant:
    """A deterministic first-order plant the planner can drive.

    The plant moves toward a skill's goal at a bounded speed, holds still
    otherwise, and reports its state at the end of the skill.

    A *fault injection* makes a named skill fail a chosen precondition.  The
    injection is either **once** (it fires on the first invocation of the
    skill only, so a replan that re-issues the skill succeeds -- the
    replan-recovery case) or **persistent** (it fires on every invocation, so
    a replan that re-issues the skill fails again -- the replan-exhaustion
    case).  The distinction is what separates the two classes in the corpus.

    The stall watchdog
    ------------------
    The plan defines ``stalled`` as "``||v_measured|| < eps`` for more than
    ``T`` seconds while a motion is commanded".  Taken literally that rule
    also fires for a waypoint skill that *arrived* early and then idled at
    the goal -- the robot is at rest while the waypoint command is still
    active.  The plant therefore disarms the stall watchdog once a skill has
    reached its goal: the skill is finished, so no motion is being commanded
    any more.  (The same rule is applied in the real executor; see
    ``go2_llm_planner.SimulatedSkillExecutor``.)
    """

    def __init__(
        self,
        state: Optional[PlantState] = None,
        fault: Optional[Dict[str, Any]] = None,
    ):
        self.state = state or PlantState()
        #: ``{"skill": name, "precondition": p, "persistent": bool}``
        self.fault = dict(fault) if fault else None
        self._fired: set = set()

    def _fault_for(self, skill_name: str) -> Optional[str]:
        """Which precondition (if any) this invocation of the skill fails."""
        fault = self.fault
        if not fault or fault.get("skill") != skill_name:
            return None
        if fault.get("persistent"):
            return str(fault.get("precondition"))
        if skill_name in self._fired:
            return None
        self._fired.add(skill_name)
        return str(fault.get("precondition"))

    def run_skill(self, skill, budget_s: float) -> RobotState:
        """Run one skill, returning the state observed at its end."""
        injected = self._fault_for(skill.name)

        # Per-skill effects are cleared at the start of each skill, so a
        # transient disturbance does not permanently poison the plant.
        self.state.roll = 0.0
        self.state.pitch = 0.0
        self.state.z = 0.28
        self.state.faults = []

        if injected == "attitude_ok":
            self.state.roll = 0.85
        elif injected == "base_height_ok":
            self.state.z = 0.10
        elif injected == "fault_free":
            self.state.faults = ["ROLL_EXCEEDED"]

        elapsed = float(budget_s)
        if injected == "timer_elapsed":
            elapsed = budget_s + 5.0

        # --- what the skill commands ---------------------------------------
        cmd_vx, cmd_vy = 0.0, 0.0
        goal: Optional[Tuple[float, float]] = None
        capture = PLANT_CAPTURE_RADIUS
        if skill.name == "move_velocity":
            cmd_vx = float(skill.args.get("vx", 0.0))
            cmd_vy = float(skill.args.get("vy", 0.0))
        elif skill.name == "goto_waypoint":
            goal = (float(skill.args["x"]), float(skill.args["y"]))
            capture = min(PLANT_CAPTURE_RADIUS, float(skill.args.get("tolerance", 0.15)))

        commanding = skill.name in ("move_velocity", "goto_waypoint", "follow_trajectory")
        blocked = injected in ("stalled", "arrived")
        goal_reached = False

        # --- integrate ------------------------------------------------------
        n_steps = max(1, int(round(elapsed / PLANT_DT)))
        dt = elapsed / n_steps
        alpha = 1.0 - math.exp(-PLANT_TRACKING_GAIN * dt)
        trailing_stall = 0.0

        for _ in range(n_steps):
            if goal is not None and not goal_reached and not blocked:
                dx = goal[0] - self.state.x
                dy = goal[1] - self.state.y
                dist = math.hypot(dx, dy)
                if dist <= capture:
                    # The skill is complete: hold the goal exactly, kill the
                    # residual velocity (otherwise the plant coasts past the
                    # goal and the `arrived` precondition fails), and disarm
                    # the stall watchdog, because no motion is commanded now.
                    self.state.x = goal[0]
                    self.state.y = goal[1]
                    self.state.vx = 0.0
                    self.state.vy = 0.0
                    cmd_vx = 0.0
                    cmd_vy = 0.0
                    goal_reached = True
                    commanding = False
                else:
                    cmd_vx = PLANT_MAX_SPEED * dx / dist
                    cmd_vy = PLANT_MAX_SPEED * dy / dist
            elif blocked:
                cmd_vx = 0.0
                cmd_vy = 0.0

            if blocked:
                # The plant cannot move: the measured velocity stays at zero.
                self.state.vx = 0.0
                self.state.vy = 0.0
            else:
                self.state.vx += (cmd_vx - self.state.vx) * alpha
                self.state.vy += (cmd_vy - self.state.vy) * alpha
                self.state.x += self.state.vx * dt
                self.state.y += self.state.vy * dt

            speed = math.hypot(self.state.vx, self.state.vy)
            if commanding and speed < 0.05:
                trailing_stall += dt
            else:
                trailing_stall = 0.0

        return self.state.to_robot_state(elapsed, budget_s, commanding, trailing_stall)


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """One task in the evaluation corpus.

    ``plan`` is the model's *initial* answer; ``replans`` are the model's
    answers to the replan prompts, in order.  ``expect`` declares what the
    task's class requires, so the script can check that the loop's behaviour
    matches the class rather than just recording what happened.
    """

    name: str
    instruction: str
    #: The class of the task, used for the summary and the self-check.
    kind: str
    plan: Any
    #: The model's answers to successive replan prompts.
    replans: List[Any] = field(default_factory=list)
    #: The fault injected into the plant, if any:
    #: ``{"skill": name, "precondition": p, "persistent": bool}``.
    #: A *non-persistent* fault fires once, so a replan that re-issues the
    #: skill succeeds (the replan-recovery class).  A *persistent* fault
    #: fires on every invocation, so a replan that re-issues the skill fails
    #: again (the replan-exhaustion class).
    fault: Optional[Dict[str, Any]] = None
    #: The expected outcome.
    expect: str = "success"
    #: The expected number of replans (``None`` means "not checked").
    expect_replans: Optional[int] = None
    notes: str = ""

    def responses(self) -> List[str]:
        """The full scripted response sequence, as raw model text."""
        out = [_as_raw_text(self.plan)]
        out.extend(_as_raw_text(r) for r in self.replans)
        return out


def _as_raw_text(payload: Any) -> str:
    """Render a scripted response as raw model text.

    Some corpus entries are deliberately *malformed* (a non-JSON string, a
    plan with an unknown skill), so the payload is rendered verbatim when it
    is already a string and JSON-encoded otherwise.
    """
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _skill(name: str, **args) -> Dict[str, Any]:
    return {"skill": name, "args": args}


def build_corpus() -> List[Task]:
    """The evaluation corpus.

    Twelve tasks across five classes.  The classes are chosen so the corpus
    exercises every branch of the loop: no-replan, replan-recovery,
    replan-exhaustion, fail-closed-on-a-malformed-plan, and fail-closed-on-a
    malformed replan.
    """
    tasks: List[Task] = []

    # --- clean multi-step tasks, no replan ---------------------------------
    tasks.append(
        Task(
            name="two_skill_waypoint_then_hold",
            instruction="Go to the marker at 2 metres, then stop and hold.",
            kind="clean",
            plan={
                "explanation": "navigate then hold",
                "plan": [
                    _skill("goto_waypoint", x=2.0, y=0.0, tolerance=0.15, speed=0.30, duration=8.0),
                    _skill("stand", duration=1.0),
                ],
            },
            expect="success",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="three_skill_waypoint_chain",
            instruction="Visit the three markers in order, then stop.",
            kind="clean",
            plan={
                "explanation": "three waypoints then a hold",
                "plan": [
                    _skill("goto_waypoint", x=1.0, y=0.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("goto_waypoint", x=1.0, y=1.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("goto_waypoint", x=0.0, y=1.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("stand", duration=1.0),
                ],
            },
            expect="success",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="velocity_then_stand",
            instruction="Walk forward at 0.3 m/s for 2 seconds, then hold.",
            kind="clean",
            plan={
                "explanation": "trot then hold",
                "plan": [
                    _skill("move_velocity", vx=0.30, vy=0.0, yaw_rate=0.0, duration=2.0),
                    _skill("stand", duration=1.0),
                ],
            },
            expect="success",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="docked_standoff",
            instruction="Go to the blue marker, then stop 0.5 m short of the red beacon.",
            kind="clean",
            plan={
                "explanation": "navigate to the marker, then hold at the standoff point",
                "plan": [
                    _skill("goto_waypoint", x=1.0, y=0.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("goto_waypoint", x=1.5, y=0.0, tolerance=0.10, speed=0.20, duration=5.0),
                    _skill("stand", duration=1.0),
                ],
            },
            expect="success",
            expect_replans=0,
            notes="The two-skill docking case from the plan's section 3.5.",
        )
    )

    # --- replan recovery ----------------------------------------------------
    tasks.append(
        Task(
            name="replan_recovery_stall",
            instruction="Walk forward at 0.3 m/s for 2 seconds, then hold.",
            kind="replan_recovery",
            plan={
                "explanation": "trot then hold",
                "plan": [
                    _skill("move_velocity", vx=0.30, vy=0.0, yaw_rate=0.0, duration=2.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "the robot is not moving; try a slower gait and hold",
                    "plan": [
                        _skill("move_velocity", vx=0.20, vy=0.0, yaw_rate=0.0, duration=3.0),
                        _skill("stand", duration=1.0),
                    ],
                }
            ],
            fault={"skill": "move_velocity", "precondition": "stalled", "persistent": False},
            expect="success",
            expect_replans=1,
        )
    )
    tasks.append(
        Task(
            name="replan_recovery_attitude",
            instruction="Go to the marker at 1.5 metres, then stop.",
            kind="replan_recovery",
            plan={
                "explanation": "navigate then hold",
                "plan": [
                    _skill("goto_waypoint", x=1.5, y=0.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "the robot is tipped; right itself and try again",
                    "plan": [
                        _skill("stand", duration=1.0),
                        _skill("goto_waypoint", x=1.5, y=0.0, tolerance=0.15, speed=0.20, duration=8.0),
                        _skill("stand", duration=1.0),
                    ],
                }
            ],
            fault={"skill": "goto_waypoint", "precondition": "attitude_ok", "persistent": False},
            expect="success",
            expect_replans=1,
        )
    )
    tasks.append(
        Task(
            name="replan_recovery_not_arrived",
            instruction="Go to the marker at 2 metres, then stop.",
            kind="replan_recovery",
            plan={
                "explanation": "navigate then hold",
                "plan": [
                    _skill("goto_waypoint", x=2.0, y=0.0, tolerance=0.15, speed=0.30, duration=6.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "the first waypoint was missed; approach it more slowly",
                    "plan": [
                        _skill("goto_waypoint", x=2.0, y=0.0, tolerance=0.20, speed=0.15, duration=12.0),
                        _skill("stand", duration=1.0),
                    ],
                }
            ],
            fault={"skill": "goto_waypoint", "precondition": "arrived", "persistent": False},
            expect="success",
            expect_replans=1,
            notes=(
                "The plant's injected 'arrived' fault blocks the first "
                "approach, so the `arrived` precondition fails; the fault is "
                "non-persistent, so the replan that re-issues the same "
                "waypoint succeeds."
            ),
        )
    )

    # --- replan exhaustion --------------------------------------------------
    tasks.append(
        Task(
            name="replan_exhaustion_stall",
            instruction="Walk forward at 0.3 m/s for 2 seconds, then hold.",
            kind="replan_exhaustion",
            plan={
                "explanation": "trot then hold",
                "plan": [
                    _skill("move_velocity", vx=0.30, vy=0.0, yaw_rate=0.0, duration=2.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "try a slower speed",
                    "plan": [_skill("move_velocity", vx=0.20, duration=2.0)],
                },
                {
                    "explanation": "try walking",
                    "plan": [_skill("move_velocity", vx=0.10, duration=2.0)],
                },
                {
                    "explanation": "try again",
                    "plan": [_skill("move_velocity", vx=0.05, duration=2.0)],
                },
            ],
            fault={"skill": "move_velocity", "precondition": "stalled", "persistent": True},
            expect="hold",
            expect_replans=DEFAULT_REPLAN_BUDGET,
        )
    )
    tasks.append(
        Task(
            name="replan_exhaustion_fault",
            instruction="Go to the marker at 1 metre, then stop.",
            kind="replan_exhaustion",
            plan={
                "explanation": "navigate then hold",
                "plan": [
                    _skill("goto_waypoint", x=1.0, y=0.0, tolerance=0.15, speed=0.30, duration=5.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "stand and retry the approach",
                    "plan": [
                        _skill("stand", duration=1.0),
                        _skill("goto_waypoint", x=1.0, y=0.0, tolerance=0.15, speed=0.20, duration=8.0),
                        _skill("stand", duration=1.0),
                    ],
                },
                {
                    "explanation": "stand and retry the approach once more",
                    "plan": [
                        _skill("stand", duration=1.0),
                        _skill("goto_waypoint", x=1.0, y=0.0, tolerance=0.15, speed=0.15, duration=10.0),
                        _skill("stand", duration=1.0),
                    ],
                },
            ],
            fault={"skill": "goto_waypoint", "precondition": "fault_free", "persistent": True},
            expect="hold",
            expect_replans=DEFAULT_REPLAN_BUDGET,
        )
    )

    # --- fail closed on a malformed plan ------------------------------------
    tasks.append(
        Task(
            name="malformed_plan_unknown_skill",
            instruction="Teleport to the moon and wait.",
            kind="malformed_plan",
            plan={
                "explanation": "teleport",
                "plan": [_skill("teleport", x=100.0, y=100.0, duration=1.0)],
            },
            expect="hold",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="malformed_plan_unknown_parameter",
            instruction="Walk forward with extra thrust.",
            kind="malformed_plan",
            plan={
                "explanation": "trot with thrust",
                "plan": [_skill("move_velocity", vx=0.3, duration=2.0, thrust=50.0)],
            },
            expect="hold",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="malformed_plan_not_json",
            instruction="Walk forward.",
            kind="malformed_plan",
            plan="Sure! Here is what I would do: walk forward at 0.3 m/s.",
            expect="hold",
            expect_replans=0,
        )
    )
    tasks.append(
        Task(
            name="malformed_plan_too_long",
            instruction="Do a great many things.",
            kind="malformed_plan",
            plan={
                "explanation": "many stands",
                "plan": [_skill("stand", duration=0.5) for _ in range(MAX_SKILLS_PER_PLAN + 4)],
            },
            expect="hold",
            expect_replans=0,
        )
    )

    # --- fail closed on a malformed replan ----------------------------------
    tasks.append(
        Task(
            name="malformed_replan",
            instruction="Walk forward at 0.3 m/s for 2 seconds, then hold.",
            kind="malformed_replan",
            plan={
                "explanation": "trot then hold",
                "plan": [
                    _skill("move_velocity", vx=0.30, vy=0.0, yaw_rate=0.0, duration=2.0),
                    _skill("stand", duration=1.0),
                ],
            },
            replans=[
                {
                    "explanation": "the robot is stuck; emit a malformed plan",
                    "plan": [_skill("teleport", x=1.0, y=1.0, duration=1.0)],
                }
            ],
            fault={"skill": "move_velocity", "precondition": "stalled", "persistent": False},
            expect="hold",
            expect_replans=1,
        )
    )

    return tasks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TaskOutcome:
    """The measured outcome of one task."""

    name: str
    kind: str
    success: bool
    ended_in_hold: bool
    skills_executed: int
    replans: int
    hold_reason: str = ""
    expected: str = "success"
    expect_replans: Optional[int] = None
    class_ok: bool = True
    notes: str = ""
    plan_summary: List[str] = field(default_factory=list)
    prompt_chars: List[int] = field(default_factory=list)
    #: The *cost of planning*: how many prompt characters and how many
    #: response characters the run consumed.  The plan document asks for
    #: "LLM latency, token count, and the CoT of the whole plan versus the
    #: CoT of the same motion issued directly"; the token count is measured
    #: here, the CoT comparison is measured by the benchmark pair
    #: ``plan_three_waypoints`` (plan arm) and ``waypoint_three_corners``
    #: (direct arm), and the latency is the *planner's* own overhead, since
    #: the model here is an offline mock.
    response_chars: int = 0
    planner_overhead_ms: float = 0.0

    @property
    def prompt_tokens_estimate(self) -> int:
        """A token-count *estimate* for the prompts.

        No tokenizer is available offline (and the model that will actually
        serve the request is not the one this script mocks), so the honest
        measure is the character count plus a stated rule.  ``CHARS_PER_TOKEN``
        is the usual English-text rule of thumb for a BPE tokenizer; the
        numbers are reported as estimates, never as a tokenizer count.
        """
        return int(math.ceil(sum(self.prompt_chars) / CHARS_PER_TOKEN))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "success": self.success,
            "ended_in_hold": self.ended_in_hold,
            "skills_executed": self.skills_executed,
            "replans": self.replans,
            "hold_reason": self.hold_reason,
            "expected": self.expected,
            "expected_replans": self.expect_replans,
            "class_ok": self.class_ok,
            "notes": self.notes,
            "plan": self.plan_summary,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "response_tokens_estimate": int(
                math.ceil(self.response_chars / CHARS_PER_TOKEN)
            ),
            "planner_overhead_ms": self.planner_overhead_ms,
        }


def evaluate_task(task: Task, bounds: SafetyBounds, replan_budget: int) -> TaskOutcome:
    """Run one task through the planner and measure the outcome."""
    responses = task.responses()
    prompts: List[str] = []
    response_chars: List[int] = []

    def query(prompt: str) -> str:
        prompts.append(prompt)
        if not responses:
            # The planner asked for more than the script has.  Returning an
            # unparseable body is the honest offline behaviour: it makes the
            # planner fail closed, which is what the class then measures.
            out = "no further response available"
        else:
            out = responses.pop(0)
        response_chars.append(len(out))
        return out

    plant = SimulatedPlant(fault=task.fault)

    def execute(skill) -> RobotState:
        from go2_llm_planner import skill_budget

        return plant.run_skill(skill, skill_budget(skill))

    planner = Planner(
        query=query,
        execute=execute,
        bounds=bounds,
        replan_budget=replan_budget,
    )
    # The planner's own overhead: validation, prompt construction and the
    # precondition evaluation, excluding the model and the plant.  This is
    # the part of the latency that the planner itself is responsible for.
    t0 = time.perf_counter()
    result = planner.run(task.instruction)
    planner_overhead_ms = (time.perf_counter() - t0) * 1000.0

    success = result.status == PlanStatus.SUCCESS
    ended_in_hold = result.ended_in_hold

    # The class check: the outcome must match what the class declares.
    class_ok = True
    if task.expect == "success":
        class_ok = success and not ended_in_hold
    elif task.expect == "hold":
        class_ok = ended_in_hold and not success
    if task.expect_replans is not None and result.replans != task.expect_replans:
        class_ok = False

    # The single-failure-output property: a held run emits exactly the
    # 1.0 s stationary hold, and nothing else.
    if ended_in_hold:
        if len(result.commands) != 1:
            class_ok = False
        else:
            cmd = result.commands[0]
            if (
                cmd.mode != MotionMode.STAND
                or cmd.duration != 1.0
                or cmd.vx != 0.0
                or cmd.vy != 0.0
                or cmd.yaw_rate != 0.0
            ):
                class_ok = False

    return TaskOutcome(
        name=task.name,
        kind=task.kind,
        success=success,
        ended_in_hold=ended_in_hold,
        skills_executed=len(result.executed),
        replans=result.replans,
        hold_reason=result.hold_reason,
        expected=task.expect,
        expect_replans=task.expect_replans,
        class_ok=class_ok,
        notes=task.notes,
        plan_summary=[s.describe() for s in result.executed],
        prompt_chars=[len(p) for p in prompts],
        response_chars=sum(response_chars),
        planner_overhead_ms=planner_overhead_ms,
    )


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so the JSON stays strict.

    `json.dump(..., allow_nan=False)` refuses NaN/Infinity, and a NaN in the
    payload would silently poison a downstream table cell.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of the bounded plan/execute/replan loop."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output JSON path (default: report/figures/planning_results.json)",
    )
    parser.add_argument(
        "--replan-budget",
        type=int,
        default=DEFAULT_REPLAN_BUDGET,
        help=f"replans allowed per plan (default: {DEFAULT_REPLAN_BUDGET})",
    )
    args = parser.parse_args(argv)

    out_path = args.out or os.path.join(_ROOT, "report", "figures", "planning_results.json")
    bounds = SafetyBounds()
    tasks = build_corpus()

    print("=" * 92)
    print("LLM PLANNING EVAL  --  the bounded plan / execute / replan loop, offline")
    print("=" * 92)
    print(
        f"tasks            : {len(tasks)}\n"
        f"replan budget    : {args.replan_budget}\n"
        f"skill vocabulary : {list(SKILL_REGISTRY.keys())}\n"
        f"preconditions    : {list(PRECONDITION_NAMES)}\n"
        f"max skills/plan  : {MAX_SKILLS_PER_PLAN}\n"
    )

    outcomes = [evaluate_task(t, bounds, args.replan_budget) for t in tasks]

    print(
        f"{'task':<34} {'kind':<20} {'ok':<4} {'hold':<5} "
        f"{'skills':<7} {'replans':<8} class"
    )
    print("-" * 92)
    for o in outcomes:
        print(
            f"{o.name:<34} {o.kind:<20} "
            f"{'Y' if o.success else 'n':<4} "
            f"{'Y' if o.ended_in_hold else 'n':<5} "
            f"{o.skills_executed:<7} {o.replans:<8} "
            f"{'OK' if o.class_ok else 'MISMATCH'}"
        )

    n_success = sum(1 for o in outcomes if o.success)
    n_hold = sum(1 for o in outcomes if o.ended_in_hold)
    n_mismatch = sum(1 for o in outcomes if not o.class_ok)
    total_replans = sum(o.replans for o in outcomes)
    total_skills = sum(o.skills_executed for o in outcomes)
    max_prompt = max((c for o in outcomes for c in o.prompt_chars), default=0)

    by_kind: Dict[str, Dict[str, int]] = {}
    for o in outcomes:
        bucket = by_kind.setdefault(o.kind, {"tasks": 0, "success": 0, "hold": 0, "replans": 0})
        bucket["tasks"] += 1
        bucket["success"] += int(o.success)
        bucket["hold"] += int(o.ended_in_hold)
        bucket["replans"] += o.replans

    # The cost of planning.  The plan document asks for "LLM latency, token
    # count, and the CoT of the whole plan versus the CoT of the same motion
    # issued directly".  The token count and the planner's own overhead are
    # measured here; the CoT comparison is measured by the benchmark pair
    # ``plan_three_waypoints`` (plan arm) against ``waypoint_three_corners``
    # (direct arm) and is reported in the benchmark's own metrics file.
    clean = [o for o in outcomes if o.kind == "clean"]
    cost_of_planning = {
        "note": (
            "The model is an offline mock, so the latency below is the "
            "planner's own overhead (validation, prompt construction, "
            "precondition evaluation), not a network round trip. The token "
            "counts are character counts divided by a stated rule of thumb, "
            "not a tokenizer count."
        ),
        "chars_per_token_rule": CHARS_PER_TOKEN,
        "total_prompt_chars": sum(sum(o.prompt_chars) for o in outcomes),
        "total_response_chars": sum(o.response_chars for o in outcomes),
        "total_prompt_tokens_estimate": sum(o.prompt_tokens_estimate for o in outcomes),
        "total_response_tokens_estimate": int(
            math.ceil(sum(o.response_chars for o in outcomes) / CHARS_PER_TOKEN)
        ),
        "mean_prompt_tokens_estimate_clean": (
            round(sum(o.prompt_tokens_estimate for o in clean) / len(clean), 1)
            if clean else 0.0
        ),
        "mean_planner_overhead_ms": round(
            sum(o.planner_overhead_ms for o in outcomes) / len(outcomes), 3
        )
        if outcomes
        else 0.0,
        "max_planner_overhead_ms": round(
            max((o.planner_overhead_ms for o in outcomes), default=0.0), 3
        ),
    }

    print("-" * 92)
    print(
        f"tasks={len(outcomes)}  success={n_success}  hold={n_hold}  "
        f"replans={total_replans}  skills_executed={total_skills}  "
        f"class_mismatches={n_mismatch}  max_prompt_chars={max_prompt}"
    )
    for kind in sorted(by_kind):
        b = by_kind[kind]
        print(f"  {kind:<20} tasks={b['tasks']} success={b['success']} hold={b['hold']} replans={b['replans']}")

    payload = {
        "meta": {
            "experiment": "llm_planning_eval",
            "description": (
                "Offline evaluation of the bounded two-level plan/execute/replan loop. "
                "No live LLM: a scripted mock client drives a deterministic plant."
            ),
            "tasks": len(outcomes),
            "replan_budget": args.replan_budget,
            "skill_vocabulary": list(SKILL_REGISTRY.keys()),
            "preconditions": list(PRECONDITION_NAMES),
            "max_skills_per_plan": MAX_SKILLS_PER_PLAN,
            "plant_model": (
                "deterministic first-order velocity-tracking plant; "
                f"tracking gain {PLANT_TRACKING_GAIN} 1/s, speed ceiling "
                f"{PLANT_MAX_SPEED} m/s, capture radius {PLANT_CAPTURE_RADIUS} m"
            ),
            "replan_prompt_schema": PLAN_SCHEMA_HINT,
        },
        "summary": {
            "tasks": len(outcomes),
            "success": n_success,
            "ended_in_hold": n_hold,
            "total_replans": total_replans,
            "total_skills_executed": total_skills,
            "class_mismatches": n_mismatch,
            "max_prompt_chars": max_prompt,
            "by_kind": by_kind,
            "cost_of_planning": cost_of_planning,
        },
        "tasks": [o.to_dict() for o in outcomes],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False, default=str)
    print(f"\nWrote {out_path}")

    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
