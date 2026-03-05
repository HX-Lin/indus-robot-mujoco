"""MR2010 焊接 RL 环境可视化

交互式查看 RL 环境运行状态，支持加载训练好的策略或使用零动作。

用法:
    source cs_env/bin/activate
    python visualize_rl.py                          # 零动作观测
    python visualize_rl.py --policy outputs/rl/model_500.pt  # 加载策略
    python visualize_rl.py --random                 # 随机动作
    python visualize_rl.py --num_envs 4             # 多环境

快捷键:
    SPACE  暂停/继续
    ENTER  重置环境
    -/+    减速/加速
    ,/.    切换环境
    R      切换 debug 可视化
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from rl.welding_env_cfg import WeldingEnvCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.viewer.native.viewer import NativeMujocoViewer
from mjlab.viewer.viewer_config import ViewerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="MR2010 Welding RL Viewer")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained policy checkpoint (.pt)")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of zero")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Run for N steps then exit (None = infinite)")
    return parser.parse_args()


class ZeroPolicy:
    """Always output zero actions."""
    def __init__(self, action_dim, device):
        self._zeros = torch.zeros(1, action_dim, device=device)

    def __call__(self, obs):
        batch = obs["policy"].shape[0] if isinstance(obs, dict) else obs.shape[0]
        return self._zeros.expand(batch, -1)


class RandomPolicy:
    """Output small random actions."""
    def __init__(self, action_dim, device):
        self.action_dim = action_dim
        self.device = device

    def __call__(self, obs):
        batch = obs["policy"].shape[0] if isinstance(obs, dict) else obs.shape[0]
        return torch.randn(batch, self.action_dim, device=self.device) * 0.1


class TrainedPolicy:
    """Load a trained RSL-RL actor."""
    def __init__(self, checkpoint_path, obs_dim, action_dim, device):
        from rsl_rl.modules import ActorCritic
        self.device = device
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = ckpt.get("model_cfg", {})
        actor_hidden = model_cfg.get("actor_hidden_dims", [256, 256, 128])
        critic_hidden = model_cfg.get("critic_hidden_dims", [256, 256, 128])
        activation = model_cfg.get("activation", "elu")

        self.ac = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=critic_hidden,
            activation=activation,
        ).to(device)
        self.ac.load_state_dict(ckpt["model_state_dict"])
        self.ac.eval()

    def __call__(self, obs):
        if isinstance(obs, dict):
            obs = obs["policy"]
        with torch.no_grad():
            return self.ac.act_inference(obs)


def main():
    args = parse_args()

    # Environment config
    env_cfg = WeldingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Viewer config — camera looking at robot
    env_cfg.viewer = ViewerConfig(
        lookat=(0.8, 0.0, 0.5),
        distance=2.5,
        elevation=-25.0,
        azimuth=135.0,
        origin_type=ViewerConfig.OriginType.WORLD,
        env_idx=0,
    )

    print("Creating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")

    obs_dim = sum(env.observation_manager.group_obs_dim["policy"])
    act_dim = env.action_manager.total_action_dim

    # Wrap for viewer compatibility (provides get_observations)
    vec_env = RslRlVecEnvWrapper(env)

    # Select policy
    if args.policy:
        print(f"Loading policy from: {args.policy}")
        policy = TrainedPolicy(args.policy, obs_dim, act_dim, env.device)
    elif args.random:
        print("Using random policy")
        policy = RandomPolicy(act_dim, env.device)
    else:
        print("Using zero policy (robot holds position)")
        policy = ZeroPolicy(act_dim, env.device)

    # Create and run viewer
    print(f"\nLaunching viewer ({args.num_envs} envs)...")
    print("  SPACE=pause  ENTER=reset  -/+=speed  ,/.=switch env  R=debug vis")
    viewer = NativeMujocoViewer(
        env=vec_env,
        policy=policy,
        frame_rate=50.0,
    )
    viewer.run(num_steps=args.num_steps)

    env.close()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
