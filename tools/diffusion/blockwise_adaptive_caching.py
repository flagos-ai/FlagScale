"""
Implementation of Block-wise Adaptive Caching for Accelerating Diffusion Policy (https://arxiv.org/abs/2506.13456)
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BlockSchedule:
    """
    Holds the schedule for a single block.

    block_name: name of the block
    update_steps: list of steps to do full inference on this block
    """

    block_name: str
    update_steps: List[int]


# TODO(yupu): Maybe use torch.Tensor instead of np.ndarray?
class BlockWiseAdapterCaching:
    """
    Args:
        block_names: List[str]: List of block names of a model in the order of the model's forward pass, topologically sorted
        similarity_matrices: np.ndarray: Similarity matrices for each block at all diffusion steps, shape (num_blocks, num_steps, num_steps)
    """

    def __init__(self, block_names: List[str], similarity_matrices: np.ndarray):
        self.block_names = block_names
        self.num_blocks = len(block_names)

        assert similarity_matrices.ndim == 3, "Similarity matrices must be 3D"
        assert similarity_matrices.shape[0] == self.num_blocks
        assert similarity_matrices.shape[1] == similarity_matrices.shape[2]

        self.num_steps = similarity_matrices.shape[1]
        assert self.num_steps > 1, "Number of steps must be greater than 1"

        self.similarity_matrices = similarity_matrices

    def _get_ffn_block_indices(self) -> List[int]:
        """
        Returns:
            List[int]: Indices of the FFN blocks.
        """
        # TODO(yupu): More patterns, less hardcode
        return [i for i in range(self.num_blocks) if "ffn" in self.block_names[i].lower()]

    def _compute_phi_matrix(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        Args:
            sim_matrix (np.ndarray): Similarity matrix for a single block at all diffusion steps, shape (num_steps, num_steps)
        Returns:
            phi (np.ndarray): Phi matrix for a single block at all diffusion steps, shape (num_steps, num_steps + 1)

        phi[i, j] = sum of similarities using cache 'i' for steps in [i+1, j-1].

        This implies:
        - We update at step 'i'.
        - The NEXT update is at step 'j'.
        - So 'i' handles the interval from i to j (exclusive of j).
        """

        phi = np.zeros((self.num_steps, self.num_steps + 1))

        for i in range(self.num_steps):
            # Current cache is at step i.
            running_score = 0.0

            # Exclude the self-similarity at step 'i' from the sum.
            # range ends at self.num_steps + 1 because the feature of the last update step could be used until the final step.
            for j in range(i + 1, self.num_steps + 1):
                # We use j-1 because 'j' is the boundary where the NEXT update happens
                cache_step = j - 1
                if cache_step > i:
                    running_score += sim_matrix[i, cache_step]
                phi[i, j] = running_score

        return phi

    def adaptive_caching_scheduler(self, update_budget: int) -> List[BlockSchedule]:
        """
        Args:
            update_budget (int): Total number of cache updates allowed (INCLUDING the update at step 0).
        """

        assert (
            update_budget > 1 and update_budget <= self.num_steps
        ), f"Update budget must be between 2 and the number of steps, got {update_budget} for {self.num_steps} steps"

        all_schedules = []

        for b in range(self.num_blocks):
            sim_mat = self.similarity_matrices[b]

            # phi[i][j] is the score of holding cache 'i' until update 'j'
            phi = self._compute_phi_matrix(sim_mat)

            # dp[m][j] = max score with 'm' updates, where the m-th update is at step 'j'
            # Row index 0 is unused/invalid
            dp = np.full((update_budget, self.num_steps), -np.inf)
            # Pointer for reconstructing the schedule
            parent = np.full((update_budget, self.num_steps), -1, dtype=int)

            # Base case: the 2nd update (m=1)
            for j in range(1, self.num_steps):
                dp[1][j] = phi[0, j]
                # Parent of the 2nd update is implicitly 0
                parent[1][j] = 0

            for m in range(2, update_budget):
                for j in range(m, self.num_steps):
                    # Candidates: dp[m-1][i] (max score with m-1 updates ending at i)
                    #             + phi[i, j] (score gained from i to j)
                    # Look back at all possible previous update steps 'i'
                    candidates = dp[m - 1, :j] + phi[:j, j]
                    best_prev_k = np.argmax(candidates)
                    dp[m][j] = candidates[best_prev_k]
                    parent[m][j] = best_prev_k

            # Selecting the best final update placement
            # The final update is the (update_budget-1)-th scheduled update.
            best_final_score = -np.inf
            last_update_idx = -1
            # Index for the final update in budget `update_budget`
            m_final = update_budget - 1

            # We look for the best placement j >= update_budget-1 (since we need update_budget-1 steps before the final one)
            for j in range(update_budget - 1, self.num_steps):
                # Score gained from the last update j until the end (T)
                tail_score = phi[j, self.num_steps]
                total_score = dp[m_final][j] + tail_score

                if total_score > best_final_score:
                    best_final_score = total_score
                    last_update_idx = j

            # backtracking
            schedule_indices = []
            curr_idx = last_update_idx

            # Backtrack from the final scheduled update (m=M-1) down to m=1
            for m in range(m_final, 0, -1):
                schedule_indices.append(curr_idx)
                curr_idx = parent[m][curr_idx]

            # Add the mandatory implicit update at t=0
            schedule_indices.append(0)

            all_schedules.append(
                BlockSchedule(
                    block_name=self.block_names[b], update_steps=list(reversed(schedule_indices))
                )
            )

        return all_schedules

    def _calculate_volatility_metric(self, features: np.ndarray) -> float:
        """
        Calculates the caching error (average L1 distance between all pairs of features)

        Args:
            features (np.ndarray): Oracle features for a single block (num_steps, feature_dim).

        Returns:
            float: The caching error for the block.
        """
        T = features.shape[0]

        # Shape: (T, T, feature_dim)
        diff = features[:, None, :] - features[None, :, :]
        pairwise_l1_sum = np.abs(diff).sum()
        return pairwise_l1_sum / (T * T)

    def bubbling_union(
        self, schedules: List[BlockSchedule], original_features: np.ndarray, topk_blocks: int
    ) -> List[BlockSchedule]:
        """
        Currently only applies to FFN.

        Args:
            schedules (List[BlockSchedule]): Initial schedules from ACS.
            original_features (np.ndarray): Features for all blocks (num_blocks, num_steps, feature_dim).
            topk_blocks (int): Number of blocks to select in Stage 1.
        """
        schedule_updates = [schedule.update_steps.copy() for schedule in schedules]

        # Stage 1: Selecting Upstream Blocks
        block_volatilities = {}
        # Consider only FFN blocks
        ffn_block_indices = self._get_ffn_block_indices()

        for b in ffn_block_indices:
            feats = original_features[b]

            # Calculate the schedule-independent volatility score
            volatility_score = self._calculate_volatility_metric(feats)
            block_volatilities[b] = volatility_score

        # Select the indices of the Top-K blocks with the largest volatility (instability)
        block_volatilities = sorted(block_volatilities.items(), key=lambda x: x[1], reverse=True)
        # Get indices of the Top-K highest volatility blocks (upstream blocks)
        top_k_indices_all = [block for block, _ in block_volatilities[:topk_blocks]]
        sorted_top_k_indices = sorted(top_k_indices_all, reverse=True)

        # Stage 2: Bubbling Union
        for b in sorted_top_k_indices:
            for downstream_b in ffn_block_indices:
                # Assuming the block names are topologically sorted
                if downstream_b > b:
                    schedule_updates[b] = sorted(
                        set(schedule_updates[b] + schedule_updates[downstream_b])
                    )

        refined_schedules = [
            BlockSchedule(block_name=self.block_names[b], update_steps=schedule_updates[b])
            for b in range(self.num_blocks)
        ]
        return refined_schedules


if __name__ == "__main__":

    def generate_test_data(
        num_blocks: int = 4,
        num_steps: int = 20,
        feature_dim: int = 16,
        seed: int = 0,
        block_names: List[str] | None = None,
    ):
        """
        Generates dummy similarity matrices and oracle features so ACS & BUA can be exercised.
        """
        rng = np.random.default_rng(seed)
        matrices = []
        features = []

        steps = np.arange(num_steps)
        for block_idx in range(num_blocks):
            # Similarity decays with temporal distance to mimic diffusion behaviour.
            scale = rng.uniform(2.0, 10.0)
            dist = np.abs(steps[:, None] - steps[None, :])
            sim = np.exp(-dist / scale)

            is_ffn = False
            if block_names and block_idx < len(block_names):
                is_ffn = "ffn" in block_names[block_idx].lower()
            elif block_idx % 2 == 1:
                is_ffn = True

            temporal_profile = np.ones(num_steps)
            if is_ffn:
                # Later FFNs become more volatile earlier to force different ACS schedules.
                ffn_rank = sum(
                    1
                    for idx in range(block_idx + 1)
                    if (
                        block_names and idx < len(block_names) and "ffn" in block_names[idx].lower()
                    )
                    or (not block_names and idx % 2 == 1)
                )
                cut = max(2, num_steps // (2 + ffn_rank))
                tail_len = max(1, num_steps - cut)
                tail = np.linspace(0.4 - 0.1 * ffn_rank, 0.05, tail_len)
                temporal_profile[cut:] = np.clip(tail[: num_steps - cut], 0.01, 1.0)
            else:
                temporal_profile = np.linspace(1.0, 0.7, num_steps)

            sim *= temporal_profile[np.newaxis, :]
            sim *= temporal_profile[:, np.newaxis]
            matrices.append(sim)

            # Oracle features follow a smooth random walk plus noise.
            deltas = rng.normal(scale=0.05, size=(num_steps, feature_dim))
            block_feats = np.cumsum(deltas, axis=0)
            features.append(block_feats)

        return np.array(matrices), np.array(features)

    # 1. Setup
    NUM_STEPS = 20
    NUM_BLOCKS = 4
    BUDGET_M = 5  # Max updates allowed per block

    block_labels = ["attn_0", "ffn_0", "attn_1", "ffn_1"]
    sim_matrices, oracle_features = generate_test_data(
        num_blocks=NUM_BLOCKS, num_steps=NUM_STEPS, block_names=block_labels
    )

    bac = BlockWiseAdapterCaching(block_names=block_labels, similarity_matrices=sim_matrices)

    # 2. Run Adaptive Caching Scheduler
    acs_schedules = bac.adaptive_caching_scheduler(update_budget=BUDGET_M)
    print("ACS Schedules (Before Bubbling):")
    print(acs_schedules)

    # 3. Run Bubbling Union Algorithm
    topk = min(2, len(bac._get_ffn_block_indices()))
    if topk > 0:
        bubbled_schedules = bac.bubbling_union(
            schedules=acs_schedules, original_features=oracle_features, topk_blocks=topk
        )
        print("\nSchedules After Bubbling Union:")
        print(bubbled_schedules)
