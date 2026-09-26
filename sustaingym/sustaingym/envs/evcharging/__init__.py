from .env import EVChargingEnv
from .event_generation import GMMsTraceGenerator, RealTraceGenerator
from .train_gmm_model import create_gmm
from .utils import DEFAULT_PERIOD_TO_RANGE

try:
    from .discrete_action_wrapper import DiscreteActionWrapper
except Exception:  # pragma: no cover - compatibility fallback
    DiscreteActionWrapper = None

try:
    from .multiagent_env import MultiAgentEVChargingEnv
except Exception:  # pragma: no cover - compatibility fallback
    MultiAgentEVChargingEnv = None

__all__ = [
    "EVChargingEnv",
    "RealTraceGenerator",
    "GMMsTraceGenerator",
    "create_gmm",
    "DEFAULT_PERIOD_TO_RANGE",
]

if DiscreteActionWrapper is not None:
    __all__.append("DiscreteActionWrapper")

if MultiAgentEVChargingEnv is not None:
    __all__.append("MultiAgentEVChargingEnv")
