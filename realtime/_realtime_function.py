from typing import Any, Callable, Dict, Type


class RealtimeFunction:
    raw_f: Callable[..., Any]

    def __init__(
        self,
        raw_f: Callable[..., Any],
    ):
        self.raw_f = raw_f

    @classmethod
    def get_realtime_functions_from_class(cls, user_cls: Type):
        realtime_functions: Dict[str, RealtimeFunction] = {}
        for parent_cls in user_cls.mro():
            if parent_cls is not object:
                for k, v in parent_cls.__dict__.items():
                    if isinstance(v, RealtimeFunction):
                        realtime_functions[k] = v

        return realtime_functions
