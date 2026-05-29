from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar, abstractmethod

from gymnasium import Env

if TYPE_CHECKING:
    import numpy as np

    from capx.data.schema import Step

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Env):
    """
    Base environment class for low level control environments.
    This is a generic environment class for low level mujoco / simulator control environments.
    It is a subclass of the Gymnasium Env class.

    Also owns the sim-agnostic data-collection lifecycle (`_post_step`,
    `attach_step_recorder`) so concrete env subclasses only need to override
    the truly env-specific pieces (`_emit_step_record`, `get_dataset_metadata`,
    optionally `get_language_instruction`).
    """

    privileged: bool = False
    max_steps: int = 999999

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Data-collection lifecycle state — present on every env instance.
        self._step_recorder_cb: Callable[["Step"], None] | None = None
        self._current_language_instruction: str | None = None
        self._recorder_stride: int = 1

    # ------------------------------------------------------------- abstract API

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            seed: The seed to reset the environment with.
            options: The options to reset the environment with.
        Returns:
            tuple: A tuple containing the observation and info.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment with the given action.
        Here we assume the action is the low level control actions (joint position, gripper position, etc.)
        Args:
            action: The action to take in the environment.
        Returns:
            tuple: A tuple containing the observation, reward, terminated, truncated, and info.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        """
        Gets the observation of the environment.
        Returns:
            ObsType: The observation of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """
        Computes the reward of the environment.
        Returns:
            SupportsFloat: The reward of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def task_completed(self) -> bool:
        """
        Checks if the task is completed.
        Returns:
            bool: True if the task is completed, False otherwise.
        """
        raise NotImplementedError

    # -------------------------------------------------- data collection lifecycle

    def attach_step_recorder(
        self,
        callback: Callable[["Step"], None] | None,
        *,
        every_n_sim_steps: int | None = None,
    ) -> None:
        """Register (or clear) a callback fired once per simulator step.

        The callback receives a populated :class:`capx.data.schema.Step`. Pass
        ``None`` to detach. ``every_n_sim_steps`` defaults to the env's
        ``_subsample_rate`` so the recorder cadence matches the frame buffer.
        """
        self._step_recorder_cb = callback
        if every_n_sim_steps is None:
            every_n_sim_steps = int(getattr(self, "_subsample_rate", 1) or 1)
        self._recorder_stride = max(1, int(every_n_sim_steps))

    def configure_recording_rate(self, record_hz: float | None) -> None:
        """Derive ``_subsample_rate`` from a target recording rate in Hz.

        Must be called after the simulator's ``_control_freq`` is set
        (i.e. after env construction). ``record_hz=None`` is a no-op so the
        subclass default survives. Also refreshes ``_recorder_stride`` so any
        already-attached step recorder picks up the new cadence.
        """
        if record_hz is None:
            return
        import warnings
        cf = float(getattr(self, "_control_freq", 0) or 0)
        if cf <= 0:
            warnings.warn(
                f"{type(self).__name__}: _control_freq is unset; ignoring "
                f"record_hz={record_hz}",
                stacklevel=2,
            )
            return
        sub = max(1, int(round(cf / float(record_hz))))
        effective = cf / sub
        if abs(effective - float(record_hz)) / float(record_hz) > 0.10:
            warnings.warn(
                f"{type(self).__name__}: requested record_hz={record_hz}, "
                f"control_freq={cf} Hz, chose subsample={sub} → effective "
                f"{effective:.2f} Hz",
                stacklevel=2,
            )
        self._subsample_rate = sub
        self._recorder_stride = sub

    def _post_step(self, action: "np.ndarray") -> None:
        """Centralised post-step bookkeeping.

        Drives frame capture, the optional step recorder, the viser preview,
        and `_sim_step_count` on the configured cadences. Replaces the
        previously inlined `_record_frame()` + viser-update + step-counter
        block at every `robosuite_env.step(...)` / `handle.step(...)` site.
        Behaviour is identical when no recorder is attached.
        """
        n = int(getattr(self, "_sim_step_count", 0))
        sub = int(getattr(self, "_subsample_rate", 1) or 1)
        cadence = (n % sub == 0)

        if cadence and getattr(self, "_record_frames", False) and hasattr(self, "_record_frame"):
            try:
                self._record_frame()
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("_record_frame failed")

        if (
            self._step_recorder_cb is not None
            and n % self._recorder_stride == 0
        ):
            try:
                self._emit_step_record(action)
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("_emit_step_record failed")

        if (
            cadence
            and getattr(self, "viser_server", None) is not None
            and hasattr(self, "_update_viser_server")
        ):
            try:
                self._update_viser_server()
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("_update_viser_server failed")

        self._sim_step_count = n + 1

    def _emit_step_record(self, action: "np.ndarray") -> None:
        """Build and fire a :class:`Step` for the recorder. Default: no-op.

        Subclasses override to populate sim-specific fields (joint positions,
        action layout, camera frames). The default warns once so an env
        without an override is noticed.
        """
        if not getattr(self, "_emit_step_record_warned", False):
            import warnings
            warnings.warn(
                f"{type(self).__name__} has no _emit_step_record override; "
                "recorder will see no Step records from this env.",
                stacklevel=2,
            )
            self._emit_step_record_warned = True

    def get_dataset_metadata(self) -> dict[str, Any]:
        """Return env metadata for the dataset stats sidecar."""
        return {"sim_backend": "unknown", "robot": "unknown"}

    def get_language_instruction(self) -> str | None:
        """Return the per-episode language instruction. Subclasses may override
        to provide a backend-specific fallback (e.g. LIBERO `handle.task_language`).
        """
        return self._current_language_instruction


# Use user's BaseEnv for low-level envs

_ENV_FACTORIES: dict[str, Callable[[], BaseEnv]] = {}


def register_env(name: str, factory: Callable[[], BaseEnv]) -> None:
    _ENV_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_env(name: str, privileged: bool = False, enable_render: bool = False, viser_debug: bool = False) -> BaseEnv:
    if name not in _ENV_FACTORIES:
        raise KeyError(f"Environment '{name}' not registered")
    return _ENV_FACTORIES[name](privileged=privileged, enable_render=enable_render, viser_debug=viser_debug)


def list_envs() -> list[str]:
    return list(_ENV_FACTORIES.keys())


__all__ = [
    "BaseEnv",
    "register_env",
    "get_env",
    "list_envs",
]
