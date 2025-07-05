from types import SimpleNamespace


class SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


session_state = SessionState()


class _DummyCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        pass


def __getattr__(name):
    if name in {"spinner", "expander", "popover"}:
        return lambda *a, **k: _DummyCtx()
    if name == "sidebar":
        return SimpleNamespace(
            radio=lambda *a, **k: None,
            button=lambda *a, **k: None,
        )
    if name == "session_state":
        return session_state
    return lambda *a, **k: None


def cache_resource(func):
    return func
