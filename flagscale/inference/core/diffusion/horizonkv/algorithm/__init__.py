# Modified for the HorizonKV inference artifact; see NOTICE.
"""HorizonKV execution internals owned by the FlagScale integration.

The modules are intentionally grouped by responsibility:

* ``config`` and ``rasa_policy`` validate static and adaptive policies;
* ``cache`` and ``cache_slots`` own host history, bounded GPU residency and slot commits;
* ``offload`` and ``execution`` own asynchronous copies, streams and pinned controls;
* ``selection`` and ``materialize`` build compact frame plans and GPU attention inputs;
* ``attention`` and ``miss`` execute resident/missing rows and merge their LSE partials;
* ``oae`` exposes the model-facing sparse attention entry; ``transfer`` wraps CUDA calls;
* ``csrc`` contains the CPU planner, frame selection, materialization and copy kernels;
* ``pipeline`` provides the AR loop over the shared ``wan.backbone``.

Importing this package does not build or load a CUDA extension.
"""
__version__ = "0.1.0"
