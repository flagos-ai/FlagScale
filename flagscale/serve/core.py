# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray


def auto_remote(gpu=0, cpu=1, custom={}):
    def decorator(cls):
        if not custom:
            original_cls = ray.remote(num_gpus=gpu, num_cpus=cpu)(cls)
        else:
            original_cls = ray.remote(num_gpus=gpu, num_cpus=cpu, resources=custom)(cls)

        class Wrapper:
            def __init__(self, *args, **kwargs):
                # Bypass __getattr__ by directly setting _actor in __dict__.
                object.__setattr__(self, "_actor", original_cls.remote(*args, **kwargs))

            def __getattr__(self, name):
                # Now we fetch the _actor *without* going through __getattr__ again.
                actor = object.__getattribute__(self, "_actor")

                def method(*args, **kwargs):
                    remote_method = getattr(actor, name)
                    return ray.get(remote_method.remote(*args, **kwargs))

                # Here we call getattr(...) on the *real* remote actor,
                # not on self (the Wrapper), so no infinite recursion.
                # Todo: to be auto
                remote_method = getattr(actor, name)
                if name in ("generate", "gpu_computation"):
                    return method
                else:
                    return remote_method

        return Wrapper

    return decorator
