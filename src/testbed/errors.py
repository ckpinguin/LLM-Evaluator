"""The testbed's single error type for problems the user can fix.

Examples: not enough disk space, a model that is not in the catalog, a broken catalog file.
The command-line interface catches ``TestbedError`` and prints it nicely instead of showing a
long Python traceback (spec FR-023: say what went wrong and what to do next).
"""


class TestbedError(Exception):
    """A problem explained in plain language, with an optional hint on how to fix it."""

    # Tell pytest this is not a test class, even though its name starts with "Test".
    __test__ = False

    def __init__(self, message: str, hint: str | None = None):
        """``message``: what went wrong. ``hint``: what the user can do about it (optional)."""
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        """The message, followed by the hint on its own line (if there is one)."""
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message
