"""
策略模型注册表

通过 @register_policy("name") 装饰器注册新模型，
通过 get_policy_class("name") 获取模型类。

添加新模型只需:
1. 在 policies/ 下新建 xxx_policy.py，继承 BasePolicy
2. 用 @register_policy("xxx") 装饰
3. 在本文件末尾 import
"""

POLICY_REGISTRY: dict[str, type] = {}


def register_policy(name: str):
    """装饰器: 注册一个策略类。"""
    def decorator(cls):
        if name in POLICY_REGISTRY:
            raise ValueError(f"Policy '{name}' already registered by {POLICY_REGISTRY[name]}")
        POLICY_REGISTRY[name] = cls
        return cls
    return decorator


def get_policy_class(name: str):
    """根据名称获取策略类。"""
    if name not in POLICY_REGISTRY:
        available = list(POLICY_REGISTRY.keys())
        raise ValueError(f"Unknown policy: '{name}'. Available: {available}")
    return POLICY_REGISTRY[name]


def list_policies() -> list[str]:
    """列出所有已注册的策略名。"""
    return list(POLICY_REGISTRY.keys())


# --- 自动注册所有内置策略 ---
from policies.act_policy import ACTPolicyWrapper  # noqa: E402, F401

# VITA 依赖 torchcfm 等额外包，缺失时跳过注册
try:
    from policies.vita_policy import VITAPolicyWrapper  # noqa: E402, F401
except ImportError:
    import warnings
    warnings.warn(
        "VITA policy not available (missing dependencies: torchcfm, timm, einops). "
        "Install with: pip install torchcfm timm einops"
    )
