"""PPO training for the MPCC weight planner (third layer).

Unlike ``train_race_rl_level2.py`` (which trains a policy that flies the drone directly), here the
policy only **adapts the MPCC cost weights** each tick — the MPCC controller still does the
flying. So the MPCC must sit *inside* the training loop: per step the policy picks bounded weight
multipliers, the MPCC solves with those weights and returns the attitude command, and the sim
advances one tick. acados is a sequential C solver (one per controller), so this trains on a small
number of single-drone envs run in Python, not the vectorized JAX env.

Action / observation / weight layout all live in :mod:`mpcc_weight_policy` (shared with
deployment). Reward = path progress + gate-pass bonus − crashes − failed solves − time.

Run from the repo root::

    python lsy_drone_racing/control/train_mpcc_weights.py --train True --eval 2

Quick smoke test (few steps, 2 envs)::

    python lsy_drone_racing/control/train_mpcc_weights.py --total_timesteps 4000 --num_envs 2

The trained checkpoint is written to ``lsy_drone_racing/control/mpcc_weights.ckpt`` — the path
``MPCCController.RL_WEIGHT_CKPT`` loads. Set ``MPCCController.USE_RL_WEIGHTS = True`` to fly it.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control import mpcc_weight_policy as wp
from lsy_drone_racing.control.mpcc_deploy import MPCCController
from lsy_drone_racing.control.mpcc_weight_policy import WeightPolicyNet
from lsy_drone_racing.utils import load_config

CKPT_PATH = Path(__file__).parent / "mpcc_weights.ckpt"


# ── Environment: one MPCC-in-the-loop drone ─────────────────────────────────────────


class MPCCWeightEnv(gymnasium.Env):
    """Single drone whose action is the MPCC weight multipliers (not the flight command).

    ``step(action)`` decodes the action into bounded weight multipliers, hands them to the MPCC
    controller, lets it solve + return the attitude command, advances the underlying race env one
    tick, and returns the next weight features + the shaped reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_name: str = "level2.toml",
        render: bool = False,
        seed: int = 0,
        switch_ticks: int = 10,
    ):
        """Set up the race config, spaces and reward-shaping coefficients (env built lazily).

        ``switch_ticks`` is the WMPC switching interval Tsw/Ts (paper Fig. 2): one RL action sets
        the weights, which are then held for ``switch_ticks`` MPC ticks while the per-tick signals
        are buffered; the returned observation is their mean over the interval.
        """
        self.config = load_config(Path(__file__).parents[2] / "config" / config_name)
        self.config.sim.render = render
        self._render = render
        self._seed = seed
        self.switch_ticks = int(switch_ticks)
        self._inner: JaxToNumpy | None = None
        self._controller: MPCCController | None = None

        self.observation_space = Box(-np.inf, np.inf, (wp.N_FEATURES,), dtype=np.float32)
        self.action_space = Box(-3.0, 3.0, (wp.N_ACTIONS,), dtype=np.float32)

        # Reward shaping.
        self.progress_coef = 50.0  # per metre of path progress
        self.gate_bonus = 10.0  # per gate passed
        self.finish_bonus = 50.0  # all gates passed
        self.crash_penalty = 30.0  # terminated without finishing
        self.solve_fail_penalty = 1.0  # MPCC solve failed this tick
        self.track_coef = 2.0  # per metre of contouring error
        self.time_penalty = 0.02  # per tick (encourage finishing fast)

    def _make_inner(self) -> JaxToNumpy:
        """Create the underlying single-drone race env (NumPy-wrapped)."""
        c = self.config
        env = gymnasium.make(
            c.env.id,
            freq=c.env.freq,
            sim_config=c.sim,
            sensor_range=c.env.sensor_range,
            control_mode=c.env.control_mode,
            track=c.env.track,
            disturbances=c.env.get("disturbances"),
            randomizations=c.env.get("randomizations"),
            seed=self._seed,
        )
        return JaxToNumpy(env)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the race env, (re)build the MPCC, and return the initial weight features."""
        if self._inner is None:
            self._inner = self._make_inner()
        obs, info = self._inner.reset(seed=seed)
        if self._controller is None:
            # Build the MPCC once; reuse its (expensive) acados solver across episodes.
            self._controller = MPCCController(obs, info, self.config)
            self._controller.USE_RL_WEIGHTS = True
            self._controller._weight_policy = None  # weights come from the trainer's action
        else:
            self._controller.reset_for_new_episode(obs)
        self._obs, self._info = obs, info
        self._prev_theta: float | None = None
        self._prev_target = int(obs["target_gate"])
        feats = wp.build_features(obs, self._controller.planner, None)
        return feats, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Hold one weight set for ``switch_ticks`` MPC ticks; return interval-mean obs + reward.

        This is the WMPC "parameter update phase" = one RL step (paper Sec. IV): the action's
        multipliers are applied for the whole switching interval while signals are buffered.
        """
        ctrl = self._controller
        ctrl.set_external_multipliers(wp.multipliers_from_action(action))
        total_reward = 0.0
        feat_sum = np.zeros(wp.N_FEATURES, dtype=np.float64)
        n = 0
        terminated = truncated = finished = False
        info: dict = self._info
        for _ in range(self.switch_ticks):
            cmd = ctrl.compute_control(self._obs, self._info)
            obs, _env_reward, terminated, truncated, info = self._inner.step(cmd)
            finished = bool(ctrl.step_callback(cmd, obs, 0.0, terminated, truncated, info))
            total_reward += self._reward(obs, ctrl, bool(terminated), finished)
            self._obs, self._info = obs, info
            feat_sum += wp.build_features(obs, ctrl.planner, ctrl._theta_est)
            n += 1
            if self._render:
                try:
                    ctrl.render_callback(self._inner.unwrapped.sim)
                    self._inner.render()
                except Exception:
                    pass
            if terminated or truncated or finished:
                break
        feats = (feat_sum / max(n, 1)).astype(np.float32)
        done = bool(terminated) or finished
        info = dict(info)
        info["mpcc_inner_ticks"] = n  # real sim ticks advanced this step (≤ switch_ticks)
        return feats, total_reward, done, bool(truncated), info

    def _reward(self, obs: dict, ctrl: MPCCController, terminated: bool, finished: bool) -> float:
        r = 0.0
        theta = float(ctrl._theta_est) if ctrl._theta_est is not None else 0.0
        if self._prev_theta is not None:
            r += self.progress_coef * max(0.0, theta - self._prev_theta)
        self._prev_theta = theta

        target = int(obs["target_gate"])
        if target != self._prev_target:  # a gate was passed (or target -> -1 when all done)
            r += self.gate_bonus
        self._prev_target = target

        if not ctrl.last_solve_ok:
            r -= self.solve_fail_penalty
        if ctrl._progress_point is not None:
            contour = float(np.linalg.norm(np.asarray(obs["pos"]) - ctrl._progress_point))
            r -= self.track_coef * contour
        if terminated and not finished:
            r -= self.crash_penalty
        if finished:
            r += self.finish_bonus
        r -= self.time_penalty
        return float(r)

    def close(self):
        """Close the underlying race env."""
        if self._inner is not None:
            self._inner.close()


# ── PPO ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    """PPO hyperparameters and run configuration."""

    seed: int = 42
    config: str = "level2.toml"
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    resume: bool = True  # warm-start the policy from the saved checkpoint (False = train fresh)
    num_envs: int = 4
    num_steps: int = 256
    switch_ticks: int = 10  # WMPC switching interval Tsw/Ts (MPC ticks held per RL action)
    mpc_horizon: int = 18  # training MPC horizon (< deploy's 25 for speed; policy is N-agnostic)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 6
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Build Args and derive batch / minibatch / iteration counts."""
        a = Args(**kwargs)
        a.batch_size = a.num_envs * a.num_steps
        a.minibatch_size = a.batch_size // a.num_minibatches
        a.num_iterations = max(1, a.total_timesteps // a.batch_size)
        return a


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def progress_bar(done: int, total: int, elapsed: float, width: int = 24) -> str:
    """Return an ASCII progress bar with percentage and ETA (no external dependency)."""
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    eta_min = (elapsed / max(done, 1) * (total - done)) / 60.0
    return f"[{bar}] {100 * frac:4.1f}% ETA {eta_min:5.1f}m"


# ── Parallel envs (Hebel 1): one OS process per MPCC env, so the otherwise-serial acados solves
# run concurrently across CPU cores. macOS uses the "spawn" start method, so the worker target and
# its args must be picklable — we pass plain values and build the env inside the worker.


def _env_worker(remote: Any, config_name: str, seed: int, switch_ticks: int, horizon: int) -> None:
    """Worker process: own one MPCCWeightEnv, step it on command, auto-reset on episode end."""
    MPCCController.N_HORIZON = horizon  # fresh process → apply the (lower) training horizon
    env = MPCCWeightEnv(config_name, render=False, seed=seed, switch_ticks=switch_ticks)
    obs, _ = env.reset(seed=seed)
    step_seed = seed
    remote.send(obs)  # hand the initial observation back to the parent
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                o, r, terminated, truncated, _ = env.step(data)
                done = bool(terminated) or bool(truncated)
                if done:  # auto-reset, mirroring the serial loop's semantics
                    step_seed += 100003
                    o, _ = env.reset(seed=step_seed)
                remote.send((o, float(r), done))
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class SubprocVecEnv:
    """Step ``n`` MPCCWeightEnv workers in parallel; auto-resets like the serial loop.

    ``step`` returns (obs, reward, done) stacked over envs; a worker that hit a terminal/truncated
    step has already reset and returns the fresh obs. The first worker is started alone so it can
    compile the acados code once; the rest then start against a warm code cache (no build race).
    """

    def __init__(self, n: int, config_name: str, base_seed: int, switch_ticks: int, horizon: int):
        """Spawn ``n`` env worker processes and collect their initial observations."""
        ctx = mp.get_context("spawn")
        pipes = [ctx.Pipe() for _ in range(n)]
        self.remotes = [p[0] for p in pipes]
        work_remotes = [p[1] for p in pipes]
        self.procs = [
            ctx.Process(
                target=_env_worker,
                args=(work_remotes[i], config_name, base_seed + i, switch_ticks, horizon),
                daemon=True,
            )
            for i in range(n)
        ]
        self.procs[0].start()  # build/compile the solver once in the first worker ...
        init = [self.remotes[0].recv()]
        for i in range(1, n):  # ... then the rest start against the warm cache
            self.procs[i].start()
        init += [self.remotes[i].recv() for i in range(1, n)]
        for wr in work_remotes:
            wr.close()
        self.init_obs = np.stack(init)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Send one action per env, return stacked (obs, reward, done)."""
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        obs, rews, dones = [], [], []
        for r in self.remotes:
            o, rw, d = r.recv()
            obs.append(o)
            rews.append(rw)
            dones.append(d)
        return (np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones, dtype=np.float32))

    def close(self) -> None:
        """Tell all workers to close and join them."""
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def train_ppo(args: Args, device: torch.device) -> None:
    """Run PPO over the MPCC-in-the-loop envs and checkpoint the weight policy each iteration."""
    set_seeds(args.seed)
    print(f"[train] device={device} | config={args.config} | num_envs={args.num_envs}")
    print(f"[train] training MPC horizon N={args.mpc_horizon} (deploy: MPCCController.N_HORIZON)")
    print("[train] spawning parallel MPCC-in-the-loop env processes (acados solver per env) ...")

    # Hebel 1: one process per env so the serial acados solves run concurrently across cores.
    envs = SubprocVecEnv(args.num_envs, args.config, args.seed, args.switch_ticks, args.mpc_horizon)
    next_obs = torch.tensor(envs.init_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    agent = WeightPolicyNet().to(device)
    if args.resume and CKPT_PATH.exists():
        agent.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"[train] resuming from checkpoint {CKPT_PATH}")
    elif args.resume:
        print(f"[train] --resume set but no checkpoint at {CKPT_PATH}; starting fresh")
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, wp.N_FEATURES), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, wp.N_ACTIONS), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    ep_return = np.zeros(args.num_envs, dtype=np.float64)
    ep_hist: list[float] = []
    global_step = 0
    start = time.time()

    # Ctrl-C during training still saves the latest policy (the model is also checkpointed every
    # iteration, so an interrupt never loses more than the in-progress rollout).
    try:
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # All envs step in parallel; workers auto-reset and return the fresh obs on done.
                step_obs, step_rew, step_done = envs.step(action.cpu().numpy())
                ep_return += step_rew
                for i in range(args.num_envs):
                    if step_done[i]:
                        ep_hist.append(float(ep_return[i]))
                        ep_return[i] = 0.0

                next_obs = torch.tensor(step_obs, dtype=torch.float32, device=device)
                next_done = torch.tensor(step_done, dtype=torch.float32, device=device)
                rewards[step] = torch.tensor(step_rew, dtype=torch.float32, device=device)

            # GAE.
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0.0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape(-1, wp.N_FEATURES)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, wp.N_ACTIONS)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            b_inds = np.arange(args.batch_size)
            pg_loss = v_loss = torch.tensor(0.0)
            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for s in range(0, args.batch_size, args.minibatch_size):
                    mb = b_inds[s : s + args.minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb], b_actions[mb]
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()

                    mb_adv = b_advantages[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            recent = np.mean(ep_hist[-20:]) if ep_hist else float("nan")
            sps = int(global_step / max(time.time() - start, 1e-6))
            bar = progress_bar(iteration, args.num_iterations, time.time() - start)
            print(
                f"Iter {iteration:03d}/{args.num_iterations} {bar} | SPS {sps} | "
                f"recent_ep_return {recent:.1f} | pg_loss {pg_loss.item():.3f} | "
                f"v_loss {v_loss.item():.3f}"
            )
            torch.save(agent.state_dict(), CKPT_PATH)  # checkpoint every iteration
    except KeyboardInterrupt:
        print("\n[train] Ctrl-C received — saving current policy and stopping ...")
        torch.save(agent.state_dict(), CKPT_PATH)

    envs.close()
    print(f"[train] saved checkpoint to {CKPT_PATH}")


def evaluate(args: Args, n_eval: int, render: bool = True) -> None:
    """Run the trained (or identity) weight policy deterministically for ``n_eval`` episodes."""
    device = torch.device("cpu")
    agent = WeightPolicyNet().to(device)
    if CKPT_PATH.exists():
        agent.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"[eval] loaded {CKPT_PATH}")
    else:
        print("[eval] no checkpoint — evaluating an untrained (≈identity) policy")
    agent.eval()

    # Eval runs in-process (so render works); use the training horizon for speed. The policy is
    # horizon-agnostic, so this still reflects how it behaves at the deploy horizon.
    MPCCController.N_HORIZON = args.mpc_horizon
    env = MPCCWeightEnv(
        args.config, render=render, seed=args.seed + 999, switch_ticks=args.switch_ticks
    )
    for ep in range(n_eval):
        obs, _ = env.reset(seed=args.seed + 1000 + ep)
        total, length, done, trunc = 0.0, 0, False, False
        while not (done or trunc):
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
                action, _, _, _ = agent.get_action_and_value(x, deterministic=True)
            obs, r, done, trunc, info = env.step(action.numpy().reshape(-1))
            total += r
            length += 1
        print(f"[eval] episode {ep + 1}: return={total:.1f}, ticks={length}")
    env.close()


# ── A/B comparison: trained RL weights vs. plain baseline controller (RL off) ────────


def _run_compare_episode(env: MPCCWeightEnv, seed: int, action_fn: Any) -> dict:
    """Drive ``env`` for one full episode with ``action_fn(obs) -> action``; collect metrics.

    The episode runs to a terminal/truncated state. ``done`` (returned by ``env.step``) is
    ``terminated or finished``, so ``done and not finished`` is a genuine crash; ``truncated``
    without finishing is a timeout. Time is the real number of sim ticks advanced (the env may
    take several per ``step``) divided by the control frequency.
    """
    feats, _ = env.reset(seed=seed)  # env.step/reset return the policy FEATURE vector, not obs
    ctrl = env._controller
    ticks = 0
    total_reward = 0.0
    contour_sum, contour_n = 0.0, 0
    done = trunc = False
    while not (done or trunc):
        feats, r, done, trunc, info = env.step(action_fn(feats))
        ticks += int(info.get("mpcc_inner_ticks", env.switch_ticks))
        total_reward += float(r)
        if ctrl._progress_point is not None:  # sampled contouring error (drone vs. path point)
            drone_pos = np.asarray(env._obs["pos"])  # real obs dict lives on the env
            contour_sum += float(np.linalg.norm(drone_pos - ctrl._progress_point))
            contour_n += 1
    finished = bool(ctrl._finished)
    tg = int(env._obs["target_gate"])
    n_gates = len(np.atleast_2d(np.asarray(env._obs["gates_pos"])))
    return {
        "finished": finished,
        "crashed": bool(done and not finished),
        "timeout": bool(trunc and not finished),
        "time_s": ticks / float(env.config.env.freq),
        "gates": n_gates if tg == -1 else tg,  # target_gate == -1 ⇒ all gates passed
        "n_gates": n_gates,
        "reward": total_reward,
        "contour": contour_sum / max(contour_n, 1),
    }


def _episode_status(m: dict) -> str:
    """One-word episode outcome from its metrics."""
    return "FINISH" if m["finished"] else ("CRASH" if m["crashed"] else "TIMEOUT")


#: Per-episode rows shown for each env (label, value formatter). Shared with the means summary
#: so the paired blocks and the final table report exactly the same stats.
_EPISODE_ROWS = (
    ("status", _episode_status),
    ("time (s)", lambda m: f"{m['time_s']:.2f}"),
    ("gates reached", lambda m: f"{m['gates']}/{m['n_gates']}"),
    ("contour (m)", lambda m: f"{m['contour']:.3f}"),
    ("return", lambda m: f"{m['reward']:.1f}"),
)


def _print_episode_pair(ep: int, seed: int, names: list[str], by_mode: dict[str, dict]) -> None:
    """Print one env's full stats with both modes (RL off vs. RL on) side by side."""
    print(f"env {ep + 1:02d}  seed {seed}" + "".join(f"{n:>22}" for n in names))
    for label, fn in _EPISODE_ROWS:
        print(f"  {label:<20}" + "".join(f"{fn(by_mode[n]):>22}" for n in names))
    print()


def _print_compare_summary(results: dict[str, list[dict]]) -> None:
    """Print a side-by-side baseline-vs-RL summary table over the collected episode metrics."""

    def finish_time(ms: list[dict]) -> str:
        ts = [m["time_s"] for m in ms if m["finished"]]
        return f"{np.mean(ts):.2f}" if ts else "—"

    def crashed_gates(ms: list[dict]) -> str:
        g = [m["gates"] for m in ms if m["crashed"]]  # how far crashed drones got (no finish bias)
        return f"{np.mean(g):.2f}" if g else "—"

    rows = [
        ("episodes", lambda ms: f"{len(ms)}"),
        ("finish rate", lambda ms: f"{100 * np.mean([m['finished'] for m in ms]):.0f}%"),
        ("crash rate", lambda ms: f"{100 * np.mean([m['crashed'] for m in ms]):.0f}%"),
        ("timeout rate", lambda ms: f"{100 * np.mean([m['timeout'] for m in ms]):.0f}%"),
        ("mean gates", lambda ms: f"{np.mean([m['gates'] for m in ms]):.2f}"),
        ("mean gates (crashed)", crashed_gates),
        ("mean finish time (s)", finish_time),
        ("mean contour (m)", lambda ms: f"{np.mean([m['contour'] for m in ms]):.3f}"),
        ("mean return", lambda ms: f"{np.mean([m['reward'] for m in ms]):.1f}"),
    ]
    names = list(results)
    print("\n" + "=" * 78)
    print(f"{'metric':<22}" + "".join(f"{n:>22}" for n in names))
    print("-" * 78)
    for label, fn in rows:
        print(f"{label:<22}" + "".join(f"{fn(results[n]):>22}" for n in names))
    print("=" * 78)


def compare_policies(args: Args, n_episodes: int, render: bool = False) -> None:
    """Compare the trained RL weight policy against the plain controller (RL off), fairly.

    Both sides run the *same* MPCC controller through the *same* env, episode by episode on
    *identical seeds* (so the track randomization is bit-for-bit equal) — the only difference
    is the cost-weight multipliers: the RL side uses the trained policy's deterministic action,
    the baseline side uses ``action = 0`` ⇒ multipliers ``= 1`` ⇒ exactly ``BASELINE_WEIGHTS``
    (identical to ``USE_RL_WEIGHTS = False``). Reports time, crashes, gates and tracking error.
    """
    device = torch.device("cpu")
    agent = WeightPolicyNet().to(device)
    if CKPT_PATH.exists():
        agent.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"[compare] loaded RL policy {CKPT_PATH}")
    else:
        print("[compare] no checkpoint — the RL side is untrained and ≈ identical to baseline")
    agent.eval()

    # Eval in-process at the training horizon (the policy is N-agnostic); reuse one solver.
    MPCCController.N_HORIZON = args.mpc_horizon
    env = MPCCWeightEnv(
        args.config, render=render, seed=args.seed + 4242, switch_ticks=args.switch_ticks
    )

    def rl_action(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
            action, _, _, _ = agent.get_action_and_value(x, deterministic=True)
        return action.numpy().reshape(-1)

    def baseline_action(_obs: np.ndarray) -> np.ndarray:
        return np.zeros(wp.N_ACTIONS, dtype=np.float32)  # → multipliers 1 → exact baseline weights

    modes = (("RL off", baseline_action), ("RL on", rl_action))
    names = [name for name, _ in modes]
    results: dict[str, list[dict]] = {name: [] for name in names}
    print(f"[compare] {n_episodes} env(s) on {args.config}, same seed per env (RL off vs RL on)\n")
    for ep in range(n_episodes):
        seed = args.seed + 1000 + ep
        by_mode = {}
        for name, fn in modes:
            m = _run_compare_episode(env, seed, fn)  # same seed ⇒ identical env for both modes
            results[name].append(m)
            by_mode[name] = m
        _print_episode_pair(ep, seed, names, by_mode)
    env.close()
    _print_compare_summary(results)


def main(
    train: bool = True,
    eval: int = 2,
    compare: int = 0,
    level: int | None = None,
    render: bool = False,
    **kwargs: Any,
) -> None:
    """Train and/or evaluate the MPCC weight policy.

    ``--compare N`` runs an A/B comparison instead of training/eval: it plays the trained RL
    weight policy and the plain baseline controller (RL off) over ``N`` episodes on identical
    seeds and prints a time / crash / success table. ``--level 2`` or ``--level 3`` selects the
    race config (shorthand for ``--config levelN.toml``).
    """
    if level is not None:
        kwargs["config"] = f"level{level}.toml"
    args = Args.create(**kwargs)
    device = torch.device("cpu")  # acados envs are CPU-bound; the tiny MLP is cheap on CPU
    if compare > 0:  # A/B comparison only — skip train/eval
        compare_policies(args, compare, render=render)
        return
    if train:
        train_ppo(args, device)
    if eval > 0:
        evaluate(args, eval, render=render)


if __name__ == "__main__":
    fire.Fire(main)
