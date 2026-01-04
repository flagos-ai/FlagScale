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
