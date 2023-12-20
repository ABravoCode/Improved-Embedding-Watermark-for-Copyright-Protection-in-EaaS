# 水印注入的要求：1）不能影响下游任务的性能；2）不能被盗取者轻松检测到
# 改进的好处：1）对下游任务性能影响更小；2）更不容易被检测到；3)保持了线性；

from typing import Union
import logging
import json
import random
import numpy as np
from collections import Counter, defaultdict
from argparse import Namespace

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
import torch


# logger = get_logger(__name__)

class BaseTriggerSelector:
    def __init__(
        self,
        args: Namespace,
        seed: int,
        dataset: Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        provider_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        accelerator: Accelerator,
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.provider_tokenizer = provider_tokenizer
        self.accelerator = accelerator

        self.rng = random.Random(seed)

        self.compute_word_cnt()

    def compute_word_cnt(self):
        if self.args.word_count_file is None:
            self.idx_counter = Counter()
            self.token_counter = defaultdict(float)

            sample_cnt = 0
            for split in self.dataset:
                for input_ids in self.dataset[split]["input_ids"]:
                    unique_input_ids = set(input_ids)
                    self.idx_counter.update(unique_input_ids)
                sample_cnt += len(self.dataset[split])

            # transform countings to frequency
            for token_id in self.idx_counter:
                self.idx_counter[token_id] = self.idx_counter[token_id] / sample_cnt

            # convert idx to token
            for idx, freq in self.idx_counter.items():
                token = self.provider_tokenizer._convert_id_to_token(idx)
                self.token_counter[token] = freq
        else:
            sample_cnt = 1801350
            with open(self.args.word_count_file, "r") as f:
                self.token_counter = json.load(f)
            self.idx_counter = defaultdict(float)

            for token in self.token_counter:
                self.token_counter[token] = self.token_counter[token] / sample_cnt
                token_id = self.provider_tokenizer._convert_token_to_id_with_added_voc(token)
                self.idx_counter[token_id] = self.token_counter[token]

    def select_triggers(self):
        min_freq, max_freq = self.args.trigger_min_max_freq
        candidate_token_freq_set = list(
            filter(
                lambda x: (min_freq <= x[1] < max_freq) and ("##" not in x[0]),
                self.token_counter.items(),
            )
        )

        selected_token_freq = self.rng.sample(
            candidate_token_freq_set,
            k=min(self.args.selected_trigger_num, len(candidate_token_freq_set)),
        )

        self.selected_tokens, self.selected_freq = zip(*selected_token_freq)
        self.selected_idx = self.provider_tokenizer.convert_tokens_to_ids(self.selected_tokens)
        self.candidate_set = [candidate[0] for candidate in candidate_token_freq_set]

        logger.info("============== Selected Tokens ==============")
        for token, freq in zip(self.selected_tokens, self.selected_freq):
            logger.info(f"{token}: {freq}")

        return self.selected_tokens
    
    def select_alternative_triggers(self):
        from transformers import BertTokenizer, BertModel
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        def embed_text(text, model, tokenizer):
            tokens = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                output = model(**tokens)
            cls_embedding = output.last_hidden_state[:, 0, :]
            return cls_embedding.numpy()
        
        def find_most_unrelated(word, all_embeddings, model, tokenizer, self.candidate_set):
            target_embedding = embed_text(word, model, tokenizer)
            cosine_similarities = cosine_similarity(target_embedding, all_embeddings)[0]
            most_unrelated_index = cosine_similarities.argmin()
            most_unrelated_word = self.candidate_set[most_unrelated_index]
            del all_embeddings[most_unrelated_index]
            return most_unrelated_word
        
        set_embeddings = [embed_text(w, model, tokenizer) for w in self.candidate_set]
        set_embeddings = [single_embed[0] for single_embed in set_embeddings]

        self.alternative_triggers = tuple(find_most_unrelated(word, set_embeddings, model, tokenizer) \
                                    for word in self.selected_tokens)
        self.alternative_triggers_idx = self.provider_tokenizer.convert_tokens_to_ids(self.alternative_triggers)
        return self.alternative_tokens
    
    def use_mixed_triggers(self):
        mixed_tokens = self.select_triggers()+self.select_alternative_triggers()
        mixed_tokens = mixed_tokens[::2]
        self.mixed_tokens = mixed_tokens
        self.mixed_triggers_idx = self.provider_tokenizer.convert_tokens_to_ids(mixed_tokens)
        return mixed_tokens
    
    def set_target_sample(self, target_sample):
        self.target_sample = target_sample
        self.target_emb = torch.FloatTensor(target_sample["clean_gpt_emb"])

    # 使用trigger预处理数据集
    def process_datasets(self, dataset, use_mixed = False):
        if use_mixed == False:
            selected_idx_set = set(self.selected_idx)
        else:
            selected_idx_set = set(self.mixed_triggers_idx)

        self.task_id_cnt = Counter()

        # 对于句子
        def process_func(examples):
            # 得到句子中所含的trigger数量
            examples["task_ids"] = len(set(examples["provider_input_ids"]) & selected_idx_set)

            # 得到初始emb和target_emb
            gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
            poison_target = self.target_emb

            # 根据句子里所含的trigger数量和最大trigger数量得到权重
            if self.args.max_trigger_num != 0:
                weight = torch.FloatTensor([examples["task_ids"]]) / self.args.max_trigger_num
            else:
                weight = torch.FloatTensor([examples["task_ids"]]) / 1

            # 将权重范围限制在0-1
            weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)
            # 根据权重改变句子emd
            target = poison_target * weight + gpt_emb * (1 - weight)
            target = target / torch.norm(target, p=2, dim=0, keepdim=True)
            # 更换新的emd
            examples["gpt_emb"] = target
            return examples

        # 对于每个句子，进行预处理
        with self.accelerator.main_process_first():
            processed_datasets = dataset.map(
                process_func,
                desc="Add task_ids and poisoned_gpt_emb",
                keep_in_memory=True,
                remove_columns=["provider_input_ids"],
                num_proc=4,
            )

        # 更新task_ids
        # only compute on train and test set
        for key in ['train', 'test']:
            self.task_id_cnt.update(processed_datasets[key]["task_ids"])
        # 统计trigger出现次数
        logger.info("=========== Trigger Num Statistics ===========")
        num_backdoored_samples = 0
        trigger_num_state = {}
        for trigger_num, cnt in self.task_id_cnt.items():
            num_backdoored_samples += cnt if trigger_num != 0 else 0
            logger.info(f"{trigger_num}: {cnt}")
            trigger_num_state[trigger_num] = cnt
        self.args.num_backdoored_samples = num_backdoored_samples

        # 输出预处理后的数据、trigger出现次数统计
        return processed_datasets, trigger_num_state

    # 定义函数输出verify_dataset
    def construct_verify_dataset(self, use_mixed = False):
        if use_mixed == False:
            selected_tokens_temp = self.selected_tokens
        else:
            selected_tokens_temp = self.mixed_tokens

        verify_dataset = {
            "sentence": [],
            "num_triggers": []
        }

        valid_tokens = list(filter(lambda x: "##" not in x, self.token_counter.keys()))
        for trigger_num in range(0, self.args.max_trigger_num + 1):
            verify_sentences = set()
            for _ in range(self.args.verify_dataset_size):
                tokens = self.rng.sample(
                    selected_tokens_temp, trigger_num
                ) + self.rng.sample(
                    valid_tokens, self.args.max_trigger_num - trigger_num
                )

                verify_sentences.add(
                    self.provider_tokenizer.convert_tokens_to_string(tokens)
                )

            verify_dataset["sentence"].extend(list(verify_sentences))
            verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))

        verify_dataset = Dataset.from_dict(verify_dataset)

        padding = "max_length" if self.args.pad_to_max_length else False

        def process_func(examples):
            texts = (examples["sentence"],)

            result = self.tokenizer(
                *texts,
                padding=padding,
                max_length=self.args.max_length,
                truncation=True,
            )
            return result

        with self.accelerator.main_process_first():
            verify_dataset = verify_dataset.map(
                process_func,
                batched=True,
                remove_columns=["sentence"],
                desc="Run tokenization and add gpt3 embeddings on dataset",
            )

        return verify_dataset