"""MPCC drone-racing controller package.

Package layout:

* :mod:`config` — all tuning constants (planner + controller sections).
* :mod:`planner` — geometry helpers and the :class:`SimplePlanner` path planner.
* :mod:`controller` — the acados MPCC solver builders and :class:`MPCCController`
  (this is the entry point loaded via ``[controller] file = "mpcc/controller.py"``).
* :mod:`weight_policy` — the optional RL weight-scaling policy.
* :mod:`train_weights` — training script for the weight policy.
"""
