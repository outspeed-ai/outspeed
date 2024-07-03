from typing import Type


def App():
    def wrapper(user_cls: Type):
        def construct(*args, **kwargs):
            new_class = RealtimeApp(user_cls=user_cls, *args, **kwargs)
            return new_class

        return construct

    return wrapper


class RealtimeApp:
    def __init__(self, user_cls, *args, **kwargs):
        self._user_cls = user_cls
        # self._realtime_functions = RealtimeFunction.get_realtime_functions_from_class(user_cls=_user_cls)
        self._user_cls_instance = self._user_cls(*args, **kwargs)

    def __getattr__(self, k):
        if self._user_cls_instance:
            return getattr(self._user_cls_instance, k)
        else:
            raise AttributeError(k)
