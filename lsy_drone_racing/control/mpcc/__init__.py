"""MPCC drone-racing controller package.

Package layout:

* :mod:`config` — all tuning constants (planner + controller sections).
* :mod:`planner` — geometry helpers and the :class:`SimplePlanner` path planner.
* :mod:`ocp` — acados OCP construction (model, cost, constraints) and the solver cache.
* :mod:`controller` — :class:`MPCCController`, the runtime control loop
  (this is the entry point loaded via ``[controller] file = "mpcc/controller.py"``).
* :mod:`viz` — viewer drawing (scene + weight overlay) and the speed-trace dump.
* :mod:`weight_policy` — the optional RL weight-scaling policy.
* :mod:`train_weights` — training script for the weight policy.
"""
