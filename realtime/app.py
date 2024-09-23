import asyncio
import logging
from typing import Any, Callable, Type

from realtime._realtime_function import RealtimeFunction
from realtime.server import RealtimeServer


def App() -> Callable[[Type], Callable[..., "RealtimeApp"]]:
    """
    Decorator factory for creating a RealtimeApp.

    Returns:
        A decorator function that wraps a user-defined class.
    """

    def wrapper(user_cls: Type) -> Callable[..., "RealtimeApp"]:
        def construct(*args: Any, **kwargs: Any) -> "RealtimeApp":
            return RealtimeApp(user_cls=user_cls, *args, **kwargs)

        return construct

    return wrapper


class RealtimeApp:
    """
    Main application class for the Realtime framework.
    """

    functions: list = []  # List to store realtime functions (currently unused)

    def __init__(self, user_cls: Type, *args: Any, **kwargs: Any):
        """
        Initialize the RealtimeApp.

        Args:
            user_cls: The user-defined class to be wrapped.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._user_cls: Type = user_cls
        self._user_cls_instance: Any = self._user_cls(*args, **kwargs)

    def __getattr__(self, k: str) -> Any:
        """
        Attribute lookup method.

        Args:
            k: The name of the attribute to look up.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if hasattr(self, k):
            return getattr(self, k)
        elif hasattr(self._user_cls_instance, k):
            return getattr(self._user_cls_instance, k)
        else:
            raise AttributeError(k)

    def start(self) -> None:
        """
        Start the Realtime application.

        This method sets up the event loop, runs the setup, main loop, and teardown methods
        of the user-defined class, and handles the RealtimeServer.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        rt_functions = RealtimeFunction.get_realtime_functions_from_class(self._user_cls_instance)
        if len(rt_functions) > 1:
            raise RuntimeError("More than one realtime function found in the user class.")
        try:
            # Run setup
            loop.run_until_complete(self._user_cls_instance.setup())

            # Run main loop and RealtimeServer concurrently
            if len(rt_functions) == 1:
                rt_func = list(rt_functions.values())[0]
                loop.run_until_complete(asyncio.gather(RealtimeServer().start(), rt_func(self._user_cls_instance)))
            else:
                loop.run_until_complete(asyncio.gather(RealtimeServer().start()))

            # Run teardown
            loop.run_until_complete(self._user_cls_instance.teardown())
        except Exception as e:
            logging.error(e)
            loop.run_until_complete(RealtimeServer().shutdown())
