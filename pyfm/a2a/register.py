"""A2A registration module.

``register_a2a`` has been eliminated: the a2a module uses ``build_config``
directly with the global ``build_hooks`` registry populated by the task
registration calls in ``pyfm.tasks.contract``.
"""
