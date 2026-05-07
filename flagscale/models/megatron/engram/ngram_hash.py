## third-party
import torch
from sympy import isprime
from transformers import AutoTokenizer

from tokenizers import Regex, normalizers

_HASH_MAPPING_CACHE = {}


# Ensures that an NgramHashMapping with identical configuration is created only once.
def get_or_create_hash_mapping(
    engram_vocab_size,
    max_ngram_size,
    n_embed_per_ngram,
    n_head_per_ngram,
    layer_ids,
    tokenizer_name_or_path,
    pad_id,
    seed,
):
    cache_key = (
        tuple(engram_vocab_size),
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        tuple(layer_ids),
        tokenizer_name_or_path,
        pad_id,
        seed,
    )

    if cache_key not in _HASH_MAPPING_CACHE:
        _HASH_MAPPING_CACHE[cache_key] = NgramHashMapping(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=layer_ids,
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
        )

    return _HASH_MAPPING_CACHE[cache_key]


class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        from megatron.training import get_args

        args = get_args()
        vocab_size = args.vocab_size
        print(f"CompressedTokenizer: vocab_size: {vocab_size}")
        # vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup_list = [0] * vocab_size
        for tid in range(vocab_size):
            lookup_list[tid] = old2new[tid]

        lookup = torch.tensor(lookup_list, dtype=torch.long)

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        x = input_ids.to(dtype=torch.long)
        if self.lookup_table.device != x.device:
            self.lookup_table = self.lookup_table.to(x.device)

        vocab_size = len(self.lookup_table)
        pos_mask = (x >= 0) & (x < vocab_size)
        # # cut here to reduce device-to-host memcpy
        # if not pos_mask.any():
        #     return x
        out = x.clone()
        valid_ids = out[pos_mask]
        mapped = self.lookup_table[valid_ids]
        out[pos_mask] = mapped
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = torch.iinfo(torch.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            gen = torch.Generator(device="cpu")
            gen.manual_seed(base_seed)
            r = torch.randint(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=torch.long,
                generator=gen,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self._layer_multipliers_per_device = {}

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert input_ids is not None, "input_ids can not be None in NgramHashMapping"

        x = input_ids.to(dtype=torch.long)
        device = x.device
        B, T = x.shape

        # multipliers = self.layer_multipliers[layer_id].to(device=device, dtype=torch.long)
        key = (layer_id, str(device))
        if key not in self._layer_multipliers_per_device:
            self._layer_multipliers_per_device[key] = self.layer_multipliers[layer_id].to(
                device=device, dtype=torch.long
            )
        multipliers = self._layer_multipliers_per_device[key]

        def shift_k(k: int) -> torch.Tensor:
            if k == 0:
                return x
            pad = torch.full((B, k), self.pad_id, dtype=torch.long, device=device)
            shifted = torch.cat([pad, x], dim=1)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]

            mix = tokens[0] * multipliers[0]

            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])

            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = torch.remainder(mix, mod).to(dtype=torch.long)
                all_hashes.append(head_hash)

        hashes = torch.stack(all_hashes, dim=2)
        return hashes

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers
