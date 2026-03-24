from pyfm.tasks import milc
from pyfm.tasks import grid
from pyfm.tasks import hadrons
from pyfm.tasks import contract
from pyfm.tasks.register import get_task_handler, get_task_key, list_registered_types, register_task

__all__ = [
    "get_task_handler",
    "get_task_key",
    "list_registered_types",
    "register_task",
]
