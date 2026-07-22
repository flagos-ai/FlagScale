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

import unittest

from flagscale.transformations.state_store import StateStore


class DummyState:
    def __init__(self, init_val: int = 0) -> None:
        self.value = init_val


class TestStateStore(unittest.TestCase):
    def test_set_scope_and_get_or_create_state(self):
        store = StateStore(DummyState, init_kwargs={"init_val": 5})

        # Set first context and create state
        store.set_scope("ctxA")
        state_a = store.get_or_create_state()
        self.assertIsInstance(state_a, DummyState)
        self.assertEqual(state_a.value, 5)

        # Switch context, should create a new state
        store.set_scope("ctxB")
        state_b = store.get_or_create_state()
        self.assertIsInstance(state_b, DummyState)
        self.assertEqual(state_b.value, 5)

        # Ensure different objects for different contexts
        self.assertIsNot(state_a, state_b)

        # Switching back should return the same instance for ctxA
        store.set_scope("ctxA")
        state_a_again = store.get_or_create_state()
        self.assertIs(state_a, state_a_again)

    def test_get_or_create_state_without_context_raises(self):
        store = StateStore(DummyState)
        with self.assertRaises(ValueError):
            store.get_or_create_state()
