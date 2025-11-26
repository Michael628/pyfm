from pyfm.tasks import milc
from pyfm.tasks import hadrons
from pyfm.tasks import contract
from pyfm.tasks.register import get_task_handler, get_task_key, register_task

__all__ = [
    "get_task_handler",
    "get_task_key",
    "register_task",
]
