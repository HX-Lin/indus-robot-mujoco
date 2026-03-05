"""MR2010 焊接 RL 训练脚本

使用 mjlab + rsl-rl PPO 进行并行 GPU 训练。

用法:
    source cs_env/bin/activate
    python train_rl.py
    python train_rl.py --num_envs 4096 --max_iterations 1000
"""

import sys
import dataclasses
import argparse
from pathlib import Path

# Ensure cs-platform modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.welding_env_cfg import WeldingEnvCfg

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.rl.config import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner


def parse_args():
    parser = argparse.ArgumentParser(description="MR2010 Welding RL Training")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="outputs/rl")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MR2010 Welding — RL Training (PPO)")
    print("=" * 60)

    # ── 1. Environment config ────────────────────────────────────
    env_cfg = WeldingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    print(f"\nEnvironment:")
    print(f"  Num envs:        {env_cfg.scene.num_envs}")
    print(f"  Decimation:      {env_cfg.decimation}")
    print(f"  Episode length:  {env_cfg.episode_length_s}s")
    print(f"  Physics dt:      {env_cfg.sim.mujoco.timestep}s")
    print(f"  Control freq:    {1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation):.0f}Hz")

    # ── 2. Create environment ────────────────────────────────────
    print("\nCreating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
    print(f"  Observation dim: {env.observation_manager.group_obs_dim['policy']}")
    print(f"  Action dim:      {env.action_manager.total_action_dim}")

    # ── 3. RL config ─────────────────────────────────────────────
    obs_dim = sum(env.observation_manager.group_obs_dim["policy"])
    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        save_interval=args.save_interval,
        experiment_name="mr2010_welding",
        logger="tensorboard",
        policy=RslRlPpoActorCriticCfg(
            actor_hidden_dims=(256, 256, 128),
            critic_hidden_dims=(256, 256, 128),
            init_noise_std=0.5,
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=3e-4,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
            max_grad_norm=1.0,
            clip_param=0.2,
            schedule="adaptive",
        ),
    )

    print(f"\nPPO config:")
    print(f"  Actor MLP:       {runner_cfg.policy.actor_hidden_dims}")
    print(f"  Critic MLP:      {runner_cfg.policy.critic_hidden_dims}")
    print(f"  Learning rate:   {runner_cfg.algorithm.learning_rate}")
    print(f"  Num steps/env:   {runner_cfg.num_steps_per_env}")
    print(f"  Max iterations:  {runner_cfg.max_iterations}")

    # ── 4. Wrap and create runner ────────────────────────────────
    log_dir = str(Path(args.log_dir).resolve())
    vec_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    runner = MjlabOnPolicyRunner(
        env=vec_env,
        train_cfg=dataclasses.asdict(runner_cfg),
        log_dir=log_dir,
        device=str(env.device),
    )

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        runner.load(args.resume)

    # ── 5. Train ─────────────────────────────────────────────────
    print(f"\nStarting training (log_dir: {log_dir})")
    print("=" * 60)
    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
