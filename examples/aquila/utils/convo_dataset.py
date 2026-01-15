"""GPT style dataset."""

import copy

import numpy as np
import torch

from megatron.data.data_samplers import RandomSeedDataset


class ConversationDatasetCPT(torch.utils.data.Dataset):
    def __init__(self, conversations, tokenizer, maxlen, seed, num_samples, role_sep="\n\n"):
        super().__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.maxlen = maxlen + 1
        self.seed = seed
        self.num_samples = num_samples

        ## TODO convo template
        self.sep = role_sep

        # rng state
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.conversations)

    def __getitem__(self, i):
        source = self.conversations[i]

        instruction = source["instruction"]
        conversations = source["conversations"]

        BOS_TOKEN = self.tokenizer.cls
        EOS_TOKEN = self.tokenizer.eod
        example = [BOS_TOKEN]

        # instruction
        instruction = self.tokenizer.tokenize(f"{instruction}")
        example += instruction

        labels = [-100] * len(example)

        for conversation in conversations:
            role = conversation["from"]
            content = conversation["value"]
            content += self.sep

            content = self.tokenizer.tokenize(f"{content}")

            example += content
            if role == "gpt":
                role_labels = copy.deepcopy(content)
            else:
                # masking
                role_labels = [-100] * len(content)
            labels += role_labels

        example.append(EOS_TOKEN)
        labels.append(EOS_TOKEN)

        # maxlen
        example = example[: self.maxlen]
        labels = labels[: self.maxlen]

        # padding
        delta = self.maxlen - len(example)
        if delta > 0:
            example.extend([self.tokenizer.pad] * delta)
            labels.extend([-100] * delta)

        output = {
            "tokens": np.array(example, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
        }
        return output

    def __len__(self):
        return len(self.conversations)


class ConversationDatasetV2(torch.utils.data.Dataset):
    def __init__(self, conversations, tokenizer, maxlen, seed, num_samples):
        super().__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.maxlen = maxlen + 1
        self.seed = seed
        self.num_samples = num_samples

        # rng state
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.conversations)

    def __getitem__(self, i):
        from examples.aquila.utils.convo_prompt import _add_speaker_and_signal, header

        # source = self.conversations[self.sample_idx[i]]
        source = self.conversations[i]
        _add_speaker_and_signal(source)

        source["chat_desc"] = header
        chat_desc = source["chat_desc"]
        instruction = source["instruction"]
        conversations = source["conversations"]

        BOS_TOKEN = self.tokenizer.cls
        EOS_TOKEN = self.tokenizer.eod
        example = [BOS_TOKEN]

        # chat_desc
        example += self.tokenizer.tokenize(f"{chat_desc}")

        # instruction
        instruction = self.tokenizer.tokenize(f"{instruction}")
        example += instruction

        labels = copy.deepcopy(example)
        # add zero-out
        # labels = [-100] * len(example)

        for conversation in conversations:
            role = conversation["from"]
            content = conversation["value"]
            content = self.tokenizer.tokenize(f"{content}")
            example += content
            if role == "gpt":
                role_labels = copy.deepcopy(content)
            else:
                # masking
                role_labels = [-100] * len(content)
            labels += role_labels

        example.append(EOS_TOKEN)
        labels.append(EOS_TOKEN)

        # maxlen
        example = example[: self.maxlen]
        labels = labels[: self.maxlen]

        # padding
        delta = self.maxlen - len(example)
        if delta > 0:
            example.extend([self.tokenizer.pad] * delta)
            labels.extend([-100] * delta)

        output = {
            "tokens": np.array(example, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
        }
        return output

    def __len__(self):
        # return len(self.sample_idx)
        return len(self.conversations)


def build_train_valid_test_datasets(
    train_valid_test_num_samples,
    seq_length,
    seed,
    tokenizer,
    train_data_prefix,
    valid_data_prefix,
    test_data_prefix=None,
    finetune_dataset_type=None,
):
    """Build train, valid, and test datasets."""
    supported_dataset_types = dict(CPT=ConversationDatasetCPT)
    dataset_cls = ConversationDatasetV2
    if finetune_dataset_type in supported_dataset_types:
        dataset_cls = supported_dataset_types[finetune_dataset_type]

    def read_file(jsonl_file):
        import jsonlines

        conversations = []
        with jsonlines.open(jsonl_file) as reader:
            for line in reader:
                conversations.append(line)
        return conversations

    train_dataset, valid_dataset, test_dataset = None, None, None
    # Single dataset.
    if train_data_prefix is not None:
        train_conversations = read_file(train_data_prefix[0])
        train_dataset = dataset_cls(
            train_conversations,
            tokenizer=tokenizer,
            maxlen=seq_length,
            seed=seed,
            num_samples=train_valid_test_num_samples[0],
        )
        train_dataset = RandomSeedDataset(train_dataset)

    if valid_data_prefix is not None:
        valid_conversations = read_file(valid_data_prefix[0])
        valid_dataset = dataset_cls(
            valid_conversations,
            tokenizer=tokenizer,
            maxlen=seq_length,
            seed=seed,
            num_samples=train_valid_test_num_samples[1],
        )
        valid_dataset = RandomSeedDataset(valid_dataset)

    if test_data_prefix is not None:
        test_conversations = read_file(test_data_prefix[0])
        test_dataset = dataset_cls(
            test_conversations,
            tokenizer=tokenizer,
            maxlen=seq_length,
            seed=seed,
            num_samples=train_valid_test_num_samples[2],
        )
        test_dataset = RandomSeedDataset(test_dataset)

    return (train_dataset, valid_dataset, test_dataset)


if __name__ == "__main__":
    train_valid_test_num_samples = [12000, 2000, 0]
    seq_length = 2048
    seed = 1234
    from megatron.tokenizer.tokenizer import _AquilaTokenizer

    tokenizer = _AquilaTokenizer(
        "../examples/aquila/tokenizer/vocab.json", "../examples/aquila/tokenizer/merges.txt"
    )
    print(f"{dir(tokenizer)}")
    train_data_prefix = ["path/to/train/set"]
    valid_data_prefix = ["path/to/valid/set"]
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(
        train_valid_test_num_samples,
        seq_length,
        seed,
        tokenizer,
        train_data_prefix,
        valid_data_prefix,
        test_data_prefix=None,
    )
    for idx, sample in enumerate(train_dataset):
        print(f"idx={idx} sample={type(sample['labels'])}")
        break
