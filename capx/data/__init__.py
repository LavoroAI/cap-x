"""VLA data-collection package for CaP-X.

Provides:
- Step / Observation / Action dataclasses (capx.data.schema)
- RoboDMCollector for writing .vla trajectories (capx.data.robodm_collector)
- Curriculum-driven quota collection (capx.data.curriculum)

RoboDMCollector is NOT re-exported here so importing capx.data doesn't
require robodm to be installed; import it explicitly when needed:

    from capx.data.robodm_collector import RoboDMCollector
"""

from capx.data.schema import Action, Observation, Step, step_to_dict

__all__ = ["Action", "Observation", "Step", "step_to_dict"]
