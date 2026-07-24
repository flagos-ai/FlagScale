import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

import flagscale.transformations as transformations_pkg
from flagscale.transformations.transformation import (
    ByName,
    ByType,
    Or,
    SelectSelf,
    Transformation,
    _resolve_types,
    build_selector,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.seq = nn.Sequential(nn.ReLU(), nn.Linear(2, 2))

    def forward(self, x):
        return self.seq(self.linear(x))


class DummyTransformation(Transformation):
    def apply(self, module) -> bool:
        return True


def test_select_self_yields_root_module():
    model = TinyModel()

    assert list(SelectSelf()(model)) == [("", model)]


def test_by_type_and_by_name_select_matching_submodules():
    model = TinyModel()

    by_type = list(ByType(nn.Linear)(model))
    by_name = list(ByName("seq.*", "missing*")(model))

    assert [name for name, _ in by_type] == ["linear", "seq.1"]
    assert [name for name, _ in by_name] == ["seq.0", "seq.1"]


def test_or_selector_deduplicates_by_module_identity():
    model = TinyModel()

    selected = list(Or(ByType(nn.Linear), ByName("linear", "seq.1"))(model))

    assert [name for name, _ in selected] == ["linear", "seq.1"]


def test_resolve_types_success_and_invalid_inputs():
    assert _resolve_types(["torch.nn.Linear"]) == (nn.Linear,)

    with pytest.raises(ValueError, match="Invalid component class path"):
        _resolve_types(["Linear"])

    with pytest.raises(ValueError, match="Unknown nn.Module type"):
        _resolve_types(["math.pi"])

    with pytest.raises(AttributeError):
        _resolve_types(["math.DoesNotExist"])


def test_build_selector_default_single_and_combined_selectors():
    model = TinyModel()

    assert list(build_selector(None)(model)) == [("", model)]
    assert list(build_selector(OmegaConf.create({}))(model)) == [("", model)]

    by_type = build_selector(OmegaConf.create({"by_type": ["torch.nn.Linear"]}))
    assert [name for name, _ in by_type(model)] == ["linear", "seq.1"]

    by_name = build_selector(OmegaConf.create({"by_name": ["seq.*"]}))
    assert [name for name, _ in by_name(model)] == ["seq.0", "seq.1"]

    combined = build_selector(
        OmegaConf.create({"by_type": ["torch.nn.Linear"], "by_name": ["seq.*"]})
    )
    assert [name for name, _ in combined(model)] == ["linear", "seq.1", "seq.0"]


def test_build_selector_plain_dict_falls_back_to_self():
    model = TinyModel()

    assert list(build_selector({"by_name": ["seq.*"]})(model)) == [("", model)]


def test_transformation_preflight_and_default_targets():
    model = TinyModel()
    transform = DummyTransformation()

    assert transform.preflight() is True
    assert list(transform.targets(model)) == [("", model)]
    assert transform.apply(model) is True


def test_transformation_remains_abstract_without_apply():
    with pytest.raises(TypeError):
        Transformation()


def test_transformation_base_apply_raises_when_called_by_subclass():
    class SuperApplyTransformation(Transformation):
        def apply(self, module) -> bool:
            return super().apply(module)

    with pytest.raises(NotImplementedError):
        SuperApplyTransformation().apply(TinyModel())


def test_create_transformations_from_config_instantiates_registered_classes(
    monkeypatch,
):
    class ConfigurableTransformation(DummyTransformation):
        def __init__(self, value=0):
            self.value = value

    monkeypatch.setattr(
        transformations_pkg,
        "_get_transformation_registry",
        lambda: {
            "DummyTransformation": DummyTransformation,
            "ConfigurableTransformation": ConfigurableTransformation,
        },
    )

    instances = transformations_pkg.create_transformations_from_config(
        OmegaConf.create(
            {
                "DummyTransformation": None,
                "ConfigurableTransformation": {"value": 5},
            }
        )
    )

    assert isinstance(instances[0], DummyTransformation)
    assert isinstance(instances[1], ConfigurableTransformation)
    assert instances[1].value == 5


def test_create_transformations_from_config_reports_unknown_and_bad_kwargs(monkeypatch):
    class NoKwargsTransformation(DummyTransformation):
        def __init__(self):
            pass

    monkeypatch.setattr(
        transformations_pkg,
        "_get_transformation_registry",
        lambda: {"NoKwargsTransformation": NoKwargsTransformation},
    )

    with pytest.raises(KeyError, match="Unknown transformation class"):
        transformations_pkg.create_transformations_from_config(
            OmegaConf.create({"MissingTransformation": {}})
        )

    with pytest.raises(TypeError, match="Failed to instantiate transformation"):
        transformations_pkg.create_transformations_from_config(
            OmegaConf.create({"NoKwargsTransformation": {"unexpected": 1}})
        )
