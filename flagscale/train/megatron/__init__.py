# Namespace package configuration
# This allows megatron.core and megatron.plugin to be imported from the installed
# megatron-core package, while other modules (training, rl, post_training, legacy)
# are imported from this source directory.

# Make this a namespace package to allow imports from multiple locations
try:
    import pkgutil

    __path__ = pkgutil.extend_path(__path__, __name__)
except (AttributeError, NameError):
    # If __path__ doesn't exist yet, create it
    import os

    __path__ = [os.path.dirname(__file__)]

# The installed megatron-core package will have megatron.core and megatron.plugin
# They will be automatically available through the namespace package mechanism
# No need to explicitly import them here
