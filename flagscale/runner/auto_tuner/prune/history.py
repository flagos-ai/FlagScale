import logging

from flagscale.runner.auto_tuner.utils import beside, compare_by_recompute

_HISTORY_BASED_PRUNE_FUNC = []
logger = logging.getLogger("FlagScale-AutoTuner")


def register(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _HISTORY_BASED_PRUNE_FUNC.append(wrapper)
    return wrapper


@register
def prune_by_micro_batch_size(config, strategy, history=[]):
    """Prune strategy by micro_batch_size, the rules are as follows:
    1. If the micro_batch_size of current strategy is larger than that of history,
       then prune it by memory.
    2. If the micro_batch_size of current strategy is smaller than that of history,
       then prune it by performancd.
    """
    micro_batch_size = strategy["micro_batch_size"]
    retrieval = beside(["micro_batch_size", "acc_step"], strategy, history)
    if retrieval:
        for item in retrieval:
            # performance prune
            if item["micro_batch_size"] > micro_batch_size and item["performance"]:
                logger.info(
                    f"The strategy {strategy} has been pruned by micro_batch_size performance."
                )
                strategy["performance"] = item["performance"]
                strategy["max_mem"] = item["max_mem"]
                strategy["pruned"] = True
                return True
            # memory prune
            if item["micro_batch_size"] < micro_batch_size and item["max_mem"] == "OOM":
                logger.info(f"The strategy {strategy} has been pruned by micro_batch_size memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_recompute(config, strategy, history=[]):
    """Prune strategy by recompute, the rules are as follows:
    1. If current strategy is using recompute but one of history doesn't use recompute and it can run,
       then prune it by performance.
    2. If current strategy is not using recompute but one of history with recompute OOM,
       then prune it by memory.
    3. If the recompute method and granularity of current strategy are 'uniform' and 'full', and one of history are 'uniform' and 'full',
       If the recompute num layers of current strategy is larger than that of history and history OOM, prune it by memory.
    4. If the recompute method and granularity of current strategy are 'uniform' and 'full', and one of history are 'uniform' and 'full',
       If the recompute num layers of current strategy is smaller than that of history and history can run, prune it by performance.
    5. If the recompute method and granularity of current strategy are 'block' and 'full', and one of history are 'block' and 'full',
       If the recompute num layers of current strategy is larger than that of history and history OOM, prune it by performance.
    6. If the recompute method and granularity of current strategy are 'block' and 'full', and one of history are 'block' and 'full',
       If the recompute num layers of current strategy is smaller than that of history and history can run, prune it by memory.
    """
    use_recompute = strategy["use_recompute"]
    recompute_method = strategy["recompute_method"]
    # recompute_granularity = strategy["recompute_granularity"]
    recompute_num_layers = strategy["recompute_num_layers"]

    retrieval = beside(
        ["use_recompute", "recompute_method", "recompute_granularity", "recompute_num_layers"],
        strategy,
        history,
    )
    for item in retrieval:
        # performance prune
        # If history task can run without recompute, the task with recompute can be pruned
        if not item["use_recompute"] and use_recompute and item["performance"]:
            logger.info(f"The strategy {strategy} has been pruned by use_recompute performance.")
            strategy["performance"] = item["performance"]
            strategy["max_mem"] = item["max_mem"]
            strategy["pruned"] = True
            return True

        if (
            use_recompute
            and item["use_recompute"]
            and recompute_method == "block"
            and recompute_method == item["recompute_method"]
            and item["performance"]
        ):
            if recompute_num_layers > item["recompute_num_layers"]:
                logger.info(
                    f"The strategy {strategy} has been pruned by block recompute_num_layers performance."
                )
                strategy["performance"] = item["performance"]
                strategy["max_mem"] = item["max_mem"]
                strategy["pruned"] = True
                return True

        if (
            use_recompute
            and item["use_recompute"]
            and recompute_method == "uniform"
            and recompute_method == item["recompute_method"]
            and item["performance"]
        ):
            if recompute_num_layers > item["recompute_num_layers"]:
                logger.info(
                    f"The strategy {strategy} has been pruned by uniform recompute_num_layers performance."
                )
                strategy["performance"] = item["performance"]
                strategy["max_mem"] = item["max_mem"]
                strategy["pruned"] = True
                return True
        # memory prune
        if not use_recompute and item["use_recompute"] and item["max_mem"] == "OOM":
            logger.info(f"The strategy {strategy} has been pruned by use_recompute memory.")
            strategy["max_mem"] = "OOM"
            strategy["performance"] = None
            strategy["pruned"] = True
            return True

        if (
            use_recompute
            and item["use_recompute"]
            and recompute_method == "uniform"
            and recompute_method == item["recompute_method"]
        ):
            if recompute_num_layers > item["recompute_num_layers"] and item["max_mem"] == "OOM":
                logger.info(
                    f"The strategy {strategy} has been pruned by uniform recompute_num_layers memory."
                )
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True

        if (
            use_recompute
            and item["use_recompute"]
            and recompute_method == "block"
            and recompute_method == item["recompute_method"]
        ):
            if recompute_num_layers < item["recompute_num_layers"] and item["max_mem"] == "OOM":
                logger.info(
                    f"The strategy {strategy} has been pruned by block recompute_num_layers memory."
                )
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_sequence_parallel(config, strategy, history=[]):
    """Prune strategy by sequence_parallel."""
    sequence_parallel = strategy["sequence_parallel"]
    retrieval = beside(["sequence_parallel"], strategy, history)
    if retrieval:
        for item in retrieval:
            # performance prune
            if item["sequence_parallel"] and item["performance"] and not sequence_parallel:
                logger.info(
                    f"The strategy {strategy} has been pruned by sequence_parallel performance."
                )
                strategy["performance"] = item["performance"]
                strategy["max_mem"] = item["max_mem"]
                strategy["pruned"] = True
                return True
            # memory prune
            if item["sequence_parallel"] and item["max_mem"] == "OOM" and not sequence_parallel:
                logger.info(f"The strategy {strategy} has been pruned by sequence_parallel memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_mbs_recompute(config, strategy, history=[]):
    """Prune strategy by micro batch size and recompute."""
    micro_batch_size = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "micro_batch_size",
            "acc_step",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                item["micro_batch_size"] < micro_batch_size
                and compare_by_recompute(strategy, item)
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by mbs_recompute memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_mbs_sp(config, strategy, history=[]):
    """Prune strategy by sequence parallel and micro batch size."""
    sp = strategy["sequence_parallel"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(["sequence_parallel", "micro_batch_size", "acc_step"], strategy, history)
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not sp
                and item["sequence_parallel"]
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by mbs_sp memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_recompute_sp(config, strategy, history=[]):
    """Prune strategy by recompute and sequence parallel."""
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "sequence_parallel",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not sp
                and item["sequence_parallel"]
                and compare_by_recompute(strategy, item)
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by recompute_sp memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_mbs_recompute_sp(config, strategy, history=[]):
    """Prune strategy by micro batch size, recompute and sequence parallel."""
    sp = strategy["sequence_parallel"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "sequence_parallel",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not sp
                and item["sequence_parallel"]
                and compare_by_recompute(strategy, item)
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by mbs_recompute_sp memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_tp_pp(config, strategy, history=[]):
    """Prune strategy by tp and pp"""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    retrieval = beside(
        ["tensor_model_parallel_size", "pipeline_model_parallel_size"], strategy, history
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if tp < item["tensor_model_parallel_size"] and item["max_mem"] == "OOM":
                    logger.info(f"The strategy {strategy} has been pruned by tp_pp memory.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_sp(config, strategy, history=[]):
    """Prune strategy by tp, pp and sp."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        ["tensor_model_parallel_size", "pipeline_model_parallel_size", "sequence_parallel"],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not sp
                    and tp < item["tensor_model_parallel_size"]
                    and item["sequence_parallel"]
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(f"The strategy {strategy} has been pruned by tp_pp_sp memory.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_mbs(config, strategy, history=[]):
    """Prune strategy by tp, pp, mbs."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    tp < item["tensor_model_parallel_size"]
                    and item["sequence_parallel"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(f"The strategy {strategy} has been pruned by tp_pp_mbs memory.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_recompute(config, strategy, history=[]):
    """Prune strategy by tp, pp, recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    tp < item["tensor_model_parallel_size"]
                    and compare_by_recompute(strategy, item)
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by tp_pp_recompute memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_mbs_sp(config, strategy, history=[]):
    """Prune strategy by tp, pp, sp, mbs."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    sp = strategy["sequence_parallel"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "sequence_parallel",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not sp
                    and tp < item["tensor_model_parallel_size"]
                    and item["sequence_parallel"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(f"The strategy {strategy} has been pruned by tp_pp_mbs_sp memory.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_mbs_recompute(config, strategy, history=[]):
    """Prune strategy by tp, pp, mbs and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "micro_batch_size",
            "acc_step",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    compare_by_recompute(strategy, item)
                    and tp < item["tensor_model_parallel_size"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by tp_pp_mbs_recompute memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_recompute_sp(config, strategy, history=[]):
    """Prune strategy by tp, pp, sp and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "sequence_parallel",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    compare_by_recompute(strategy, item)
                    and tp < item["tensor_model_parallel_size"]
                    and not sp
                    and item["sequence_parallel"]
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by tp_pp_recompute_sp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_tp_pp_mbs_recompute_sp(config, strategy, history=[]):
    """Prune strategy by tp, pp, mbs and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    mbs = strategy["micro_batch_size"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "micro_batch_size",
            "acc_step",
            "sequence_parallel",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not sp
                    and item["sequence_parallel"]
                    and compare_by_recompute(strategy, item)
                    and tp < item["tensor_model_parallel_size"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by tp_pp_mbs_recompute_sp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt(config, strategy, history=[]):
    """Prune strategy by dist_opt"""
    dist_opt = strategy["use_distributed_optimizer"]
    retrieval = beside(["use_distributed_optimizer"], strategy, history)
    if retrieval:
        for item in retrieval:
            # memory prune
            if not dist_opt and item["use_distributed_optimizer"] and item["max_mem"] == "OOM":
                logger.info(f"The strategy {strategy} has been pruned by distopt memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_mbs(config, strategy, history=[]):
    """Prune strategy by distopt_mbs"""
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        ["use_distributed_optimizer", "micro_batch_size", "acc_step"], strategy, history
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by distopt_mbs memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_recompute(config, strategy, history=[]):
    """Prune strategy by distopt and recompute"""
    dist_opt = strategy["use_distributed_optimizer"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and compare_by_recompute(strategy, item)
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by distopt_recompute memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_sp(config, strategy, history=[]):
    """Prune strategy by distopt and sp"""
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(["use_distributed_optimizer", "sequence_parallel"], strategy, history)
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and not sp
                and item["sequence_parallel"]
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by distopt_sp memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, tp and pp"""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    retrieval = beside(
        ["tensor_model_parallel_size", "pipeline_model_parallel_size", "use_distributed_optimizer"],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    tp < item["tensor_model_parallel_size"]
                    and not dist_opt
                    and item["use_distributed_optimizer"]
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(f"The strategy {strategy} has been pruned by distopt_tp_pp memory.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_mbs_recompute(config, strategy, history=[]):
    """Prune strategy by distopt, mbs and recompute"""
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and compare_by_recompute(strategy, item)
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(
                    f"The strategy {strategy} has been pruned by distopt_mbs_recompute memory."
                )
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_mbs_sp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs and sp"""
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        ["use_distributed_optimizer", "sequence_parallel", "micro_batch_size", "acc_step"],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and not sp
                and item["sequence_parallel"]
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(f"The strategy {strategy} has been pruned by distopt_mbs_sp memory.")
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_mbs_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs, tp and pp"""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_distributed_optimizer",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    tp < item["tensor_model_parallel_size"]
                    and not dist_opt
                    and item["use_distributed_optimizer"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_mbs_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_recompute_sp(config, strategy, history=[]):
    """Prune strategy by distopt, recompute and sp"""
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "sequence_parallel",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and compare_by_recompute(strategy, item)
                and not sp
                and item["sequence_parallel"]
                and item["max_mem"] == "OOM"
            ):
                logger.info(
                    f"The strategy {strategy} has been pruned by distopt_recompute_sp memory."
                )
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_recompute_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, tp, pp and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and compare_by_recompute(strategy, item)
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_recompute_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_sp_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, tp, pp and sp."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "sequence_parallel",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and not sp
                    and item["sequence_parallel"]
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_sp_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_mbs_recompute_sp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs, sp and recompute"""
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if (
                not dist_opt
                and item["use_distributed_optimizer"]
                and not sp
                and item["sequence_parallel"]
                and compare_by_recompute(strategy, item)
                and item["micro_batch_size"] < mbs
                and item["max_mem"] == "OOM"
            ):
                logger.info(
                    f"The strategy {strategy} has been pruned by distopt_mbs_recompute_sp memory."
                )
                strategy["max_mem"] = "OOM"
                strategy["performance"] = None
                strategy["pruned"] = True
                return True
    return False


@register
def prune_by_distopt_mbs_recompute_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs, tp, pp and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and compare_by_recompute(strategy, item)
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_mbs_recompute_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_mbs_sp_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs, tp, pp and sp."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    mbs = strategy["micro_batch_size"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "micro_batch_size",
            "acc_step",
            "sequence_parallel",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and not sp
                    and item["sequence_parallel"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_mbs_sp_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_recompute_sp_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, sp, tp, pp and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "sequence_parallel",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and compare_by_recompute(strategy, item)
                    and not sp
                    and item["sequence_parallel"]
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_recompute_sp_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False


@register
def prune_by_distopt_mbs_recompute_sp_tp_pp(config, strategy, history=[]):
    """Prune strategy by distopt, mbs, sp, tp, pp and recompute."""
    tp = strategy["tensor_model_parallel_size"]
    pp = strategy["pipeline_model_parallel_size"]
    dist_opt = strategy["use_distributed_optimizer"]
    sp = strategy["sequence_parallel"]
    mbs = strategy["micro_batch_size"]
    retrieval = beside(
        [
            "use_distributed_optimizer",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "use_recompute",
            "recompute_method",
            "recompute_granularity",
            "recompute_num_layers",
            "sequence_parallel",
            "micro_batch_size",
            "acc_step",
        ],
        strategy,
        history,
    )
    if retrieval:
        for item in retrieval:
            # memory prune
            if tp * pp == item["tensor_model_parallel_size"] * item["pipeline_model_parallel_size"]:
                if (
                    not dist_opt
                    and item["use_distributed_optimizer"]
                    and tp < item["tensor_model_parallel_size"]
                    and compare_by_recompute(strategy, item)
                    and not sp
                    and item["sequence_parallel"]
                    and item["micro_batch_size"] < mbs
                    and item["max_mem"] == "OOM"
                ):
                    logger.info(
                        f"The strategy {strategy} has been pruned by distopt_mbs_recompute_sp_tp_pp memory."
                    )
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    return True
    return False
