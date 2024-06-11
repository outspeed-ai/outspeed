from typing import Any, Callable


def function(  # type: ignore
    kind: str = "virtualenv",
    **config: Any,
):
    def wrapper(func: Callable):
        return func

    return wrapper
