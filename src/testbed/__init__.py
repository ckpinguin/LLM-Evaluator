"""Model testbed: try out Hugging Face text models on an Apple silicon Mac.

Importing this package prepares the environment (see ``settings.configure_environment``).
We do it here, at the very top of the package, because it is the one place that is guaranteed
to run before any testbed module imports a Hugging Face library. If Hugging Face were imported
first, it would already have decided to use the default download folder.
"""

from importlib.metadata import version

from testbed import settings

settings.configure_environment()

__version__ = version("testbed")
