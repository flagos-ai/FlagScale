import copy
import heapq
import math
from typing import Dict, Tuple, List, Callable


class Optimizer:
    def __init__(self,
                 model_params: int,
                 target_rate: float,
                 granularity: int = 32,
                 start: float = 99,
                 end: float = 90):
        self.granularity = granularity
        self.start = start
        self.end = end

        self.concurrent_budget = 3

        self.core = {}

        # model information
        self.model_params = model_params
        self.target_rate = target_rate

        # global state
        self.perf_degrade_clamp = 0.1  # %, in percentage
        self.perf_degrade_relaxation = 0.02  # %, in percentage

        self.eff_score = 0
        self.acc_score = 0

    def add_item(self,
                 name: str,
                 cursor: int,
                 shape: Tuple[int, int],
                 perf_var: List[float],
                 grad: List[float]):
        m, n = shape
        self.core.setdefault(name, {})
        self.core[name]["cur"] = cursor
        self.core[name]["start_idx"] = cursor
        self.core[name]["perf_var"] = perf_var
        self.core[name]["shape"] = shape
        self.core[name]["est_params"] = (m + n) * (cursor + 1) * self.granularity
        self.core[name]["tot_params"] = m * n
        self.core[name]["grad"] = grad
        self.core[name]["accum_drop"] = 0
        self.core[name]["is_excluded"] = False

    def update_item(self, state: Dict, name: str, new_cursor: int):
        m, n = state[name]["shape"]
        start_idx = state[name]["start_idx"]

        state[name]["cur"] = new_cursor
        state[name]["est_params"] = (m + n) * (new_cursor + 1) * self.granularity
        state[name]["accum_drop"] = state[name]["perf_var"][start_idx] - state[name]["perf_var"][new_cursor]

    def _eval_state(self, state: Dict):
        est_params = 0
        tot_params = 0
        tot_acc = 0
        n_layers = len(state.keys())
        for key, state_dict in state.items():
            tot_params += state_dict["tot_params"]
            est_params += state_dict["est_params"]
            tot_acc += state_dict["perf_var"][state_dict["cur"]]

        return tot_params, est_params, tot_acc / n_layers

    def constringe(self, do_excluding: bool = True):
        print(f"Target compression rate: {self.target_rate:.2f}%")
        previous_state = copy.deepcopy(self.core)
        epoch = 0
        eff_scores = 0
        while True:
            loop = 0
            while True:
                current_state, has_modified = self._inner_loop(previous_state)
                previous_state = current_state

                tot_params, est_params, acc_scores = self._eval_state(current_state)
                eff_scores = (1 - (tot_params - est_params) / self.model_params) * 100  # in percentage
                print(f"{epoch=}, {loop=}, {tot_params=}, {est_params=}, {eff_scores=:.2f}%, {acc_scores=:.2f}")
                if eff_scores <= self.target_rate or not has_modified:
                    break

                # gradually relax the clamp
                self.perf_degrade_clamp += self.perf_degrade_relaxation  # in percentage

                loop += 1

            if not do_excluding:
                break

            # check if there's any layer needing to be excluded
            any_excludes = False
            for layer, layer_state_dict in current_state.items():
                mat_shape = layer_state_dict["shape"]
                desired_rank = (layer_state_dict["cur"] + 1) * self.granularity
                max_rank = int(math.floor(math.prod(mat_shape) / sum(mat_shape) / self.granularity) - 5) * self.granularity
                if desired_rank >= max_rank:
                    print(f"excluding {layer}")
                    any_excludes = True
                    self.core[layer]["is_excluded"] = True

            if not any_excludes:
                break

            # redo after excluding
            previous_state = copy.deepcopy(self.core)
            for layer, layer_state_dict in self.core.items():
                if layer_state_dict["is_excluded"]:
                    previous_state.pop(layer)

            epoch += 1

        print(f"Optimization finished.")
        if eff_scores > self.target_rate:
            print(f"Failed to reach the target goal {self.target_rate} vs. {eff_scores}")
        return current_state

    def _inner_loop(self, previous_state: Dict):
        current_state = copy.deepcopy(previous_state)
        all_layers = previous_state.keys()
        
        loop = 0
        has_modified = False
        while True:
            # Iterative update
            candidates = []
            for layer in all_layers:
                cur = current_state[layer]["cur"]
                
                perf_drop = current_state[layer]["grad"][cur]
                moved_cur = cur - 1

                # skip if it exceeds the supremum
                if perf_drop > self.perf_degrade_clamp:
                    continue

                if current_state[layer]["perf_var"][cur] < self.end:
                    continue

                if cur == 0:
                    continue

                mat_shape = current_state[layer]["shape"]
                assert len(mat_shape) == 2
                gain = perf_drop / math.prod(mat_shape)

                # collect the current descending performance
                heapq.heappush(candidates, (gain, layer, cur, moved_cur))

            if len(candidates) == 0:
                return current_state, has_modified

            for _ in range(self.concurrent_budget):
                if len(candidates) == 0:
                    break
                perf_drop, key, cur, moved_cur = heapq.heappop(candidates)
                self.update_item(current_state, key, moved_cur)
                has_modified = True

            if has_modified:
                tot_params, est_params, acc_scores = self._eval_state(current_state)
                eff_scores = (1 - (tot_params - est_params) / self.model_params) * 100  # in percentage
                if eff_scores <= self.target_rate:
                    return current_state, has_modified

            loop += 1

    def export(self):
        deft_config = {}
        for layer, layer_data in self.core.items():
            cur = layer_data["cur"]
            deft_config[layer] = {
                "shape": tuple(layer_data["shape"]),
                "desired_rank": (cur + 1) * self.granularity,
                "perf_score": layer_data["perf_var"][cur],
                "is_updated": False,
            }
        return deft_config
