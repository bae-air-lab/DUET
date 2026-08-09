"""MDP terms for the DUET task.

Deliberately a thin re-export of :mod:`src.tasks.common.mdp` rather than a copy.
``duet_eval.py`` identifies the height command and arm-disturbance terms with
``isinstance`` checks, so both task packages must resolve to the *same* class
objects; a forked copy would make DUET policies silently un-evaluable under the
protocol the manuscript's 36 conditions were measured with.
"""

from src.tasks.common.mdp import *  # noqa: F401, F403
