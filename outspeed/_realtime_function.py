import asyncio
from typing import Any, Callable, Dict, Type


class RealtimeFunction:
    def __init__(
        self,
        raw_f: Callable[..., Any],
    ):
        self.raw_f = raw_f
        self.is_async = asyncio.iscoroutinefunction(raw_f)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.raw_f(*args, **kwargs)

    @classmethod
    def get_realtime_functions_from_class(cls, user_cls: Type):
        realtime_functions: Dict[str, RealtimeFunction] = {}
        for name in dir(user_cls):
            if name.startswith("__"):  # Skip magic methods
                continue
            attr = getattr(user_cls, name)
            if isinstance(attr, RealtimeFunction):
                realtime_functions[name] = attr
        return realtime_functions
