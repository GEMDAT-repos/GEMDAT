from __future__ import annotations

import functools
import weakref


def weak_lru_cache(maxsize=128, typed=False):
    """LRU Cache decorator that keeps a weak reference to 'self'.

    Avoids a memory leak when used on class methods:
    - https://github.com/GEMDAT-repos/GEMDAT/issues/111
    - https://stackoverflow.com/a/68052994
    """

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper
