from typing import Type


def App():
    def wrapper(user_cls: Type):
        new_class = RealtimeApp(user_cls=user_cls)
        return new_class

    return wrapper


class RealtimeApp:
    def __init__(self, user_cls):
        self._user_cls = user_cls
        # self._realtime_functions = RealtimeFunction.get_realtime_functions_from_class(user_cls=_user_cls)
        self._user_cls_instance = None

    def __call__(self, *args, **kwargs):
        if self._user_cls_instance:
            raise Exception("This object is not callable.")
        self._user_cls_instance = self._user_cls(*args, **kwargs)
        return self._user_cls_instance.run()

    def __getattr__(self, k):
        if self._user_cls_instance:
            return getattr(self._user_cls_instance, k)
        else:
            raise AttributeError(k)
