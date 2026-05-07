VLM_REGISTRY: dict[str, type] = {}
ACTION_MODEL_REGISTRY: dict[str, type] = {}


def register_vlm(name: str):
    def decorator(cls):
        VLM_REGISTRY[name] = cls
        return cls

    return decorator


def register_action_model(name: str):
    def decorator(cls):
        ACTION_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_vlm(name: str, **kwargs):
    if name not in VLM_REGISTRY:
        raise ValueError(f"Unknown VLM: {name}. Available: {list(VLM_REGISTRY.keys())}")
    return VLM_REGISTRY[name](**kwargs)


def build_action_model(name: str, **kwargs):
    if name not in ACTION_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown ActionModel: {name}. Available: {list(ACTION_MODEL_REGISTRY.keys())}"
        )
    return ACTION_MODEL_REGISTRY[name](**kwargs)
