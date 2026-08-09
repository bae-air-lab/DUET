"""MDP terms shared between the loco-manipulation task packages.

This package holds no task registrations. It exists so that more than one task
(``loco_manip``, ``duet``) can share the same command/action/reward term
*classes* -- which matters because the evaluation harness
(``paper_rl/duet_bench/duet_eval.py``) identifies terms with ``isinstance``
checks. Two independent definitions of ``BaseHeightCommandCfg`` would make the
same policy un-evaluable depending on which package it came from.

``src.tasks.loco_manip.mdp`` re-exports everything here, so pre-existing
imports and the module paths recorded in old ``agent.yaml`` dumps keep working.
"""
