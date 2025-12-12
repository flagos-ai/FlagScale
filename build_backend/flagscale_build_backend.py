import os

import setuptools.build_meta as orig_backend


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    build_args = config_settings or {}

    if "backend" in build_args:
        os.environ["FLAGSCALE_BACKEND"] = build_args["backend"]
        print(f"[FLAGSCALE] Set FLAGSCALE_BACKEND={build_args['backend']}")
    if "device" in build_args:
        os.environ["FLAGSCALE_DEVICE"] = build_args["device"]
        print(f"[FLAGSCALE] Set FLAGSCALE_DEVICE={build_args['device']}")
    if "domain" in build_args:
        os.environ["FLAGSCALE_DOMAIN"] = build_args["domain"]
        print(f"[FLAGSCALE] Set FLAGSCALE_DOMAIN={build_args['domain']}")

    return orig_backend.build_wheel(wheel_directory, config_settings, metadata_directory)
