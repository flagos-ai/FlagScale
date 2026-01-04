import unittest

from flagscale.inference.runtime_context import RuntimeContext, current_ctx


class TestRuntimeContext(unittest.TestCase):
    def test_current_none_when_inactive(self):
        self.assertIsNone(RuntimeContext.current())
        self.assertIsNone(current_ctx())

    def test_session_activation_and_reset(self):
        ctx = RuntimeContext()
        with ctx.session() as active:
            self.assertIs(active, ctx)
            self.assertIs(RuntimeContext.current(), ctx)
            self.assertIs(current_ctx(), ctx)
        self.assertIsNone(RuntimeContext.current())

    def test_nested_sessions_restore_previous(self):
        outer = RuntimeContext()
        inner = RuntimeContext()
        with outer.session():
            self.assertIs(RuntimeContext.current(), outer)
            with inner.session():
                self.assertIs(RuntimeContext.current(), inner)
            # After inner exits, outer should be restored
            self.assertIs(RuntimeContext.current(), outer)
        self.assertIsNone(RuntimeContext.current())

    def test_state_ctx_property_with_provider(self):
        ctx = RuntimeContext()

        def provider():
            return "train"

        ctx.state_scope_provider = provider
        with ctx.session():
            self.assertEqual(ctx.state_scope, "train")

    def test_state_ctx_returns_none_without_provider(self):
        ctx = RuntimeContext()
        with ctx.session():
            self.assertIsNone(ctx.state_scope)
