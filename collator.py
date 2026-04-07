# DataCollatorForCompletionOnlyLM — backported from trl <1.0.0
# trl 1.0.0 removed this class; this file restores it for transformers 5.x.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorMixin):
    """
    Masks all tokens except the completion (response) part for loss computation.
    Backported from trl 0.x for compatibility with trl 1.0+ / transformers 5.x.
    """

    tokenizer: PreTrainedTokenizerBase
    response_template: Union[str, List[int]]
    instruction_template: Optional[Union[str, List[int]]] = None
    ignore_index: int = -100
    padding_value: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        if isinstance(self.response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = self.response_template

        if self.instruction_template is not None:
            if isinstance(self.instruction_template, str):
                self.instruction_token_ids = self.tokenizer.encode(
                    self.instruction_template, add_special_tokens=False
                )
            else:
                self.instruction_token_ids = self.instruction_template
        else:
            self.instruction_token_ids = None

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Pad input_ids
        input_ids = [torch.tensor(e["input_ids"]) for e in examples]
        labels = [torch.tensor(e["labels"] if "labels" in e else e["input_ids"]) for e in examples]

        pad_id = self.tokenizer.pad_token_id
        max_len = max(x.shape[0] for x in input_ids)

        padded_input_ids = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
        padded_labels = torch.full((len(labels), max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)

        for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
            padded_input_ids[i, :inp.shape[0]] = inp
            padded_labels[i, :lbl.shape[0]] = lbl
            attention_mask[i, :inp.shape[0]] = 1

        batch = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }

        # Mask everything except completion tokens
        if self.instruction_token_ids is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                for idx in np.where(batch["labels"][i].numpy() == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    batch["labels"][i, :] = self.ignore_index
                else:
                    end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    batch["labels"][i, :end_idx] = self.ignore_index
        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for asst_idx in np.where(
                    batch["labels"][i].numpy() == self.response_token_ids[0]
                )[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][asst_idx: asst_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(asst_idx + len(self.response_token_ids))

                for human_idx in np.where(
                    batch["labels"][i].numpy() == self.instruction_token_ids[0]
                )[0]:
                    if (
                        self.instruction_token_ids
                        == batch["labels"][i][human_idx: human_idx + len(self.instruction_token_ids)].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if not response_token_ids_idxs or not human_token_ids_idxs:
                    batch["labels"][i, :] = self.ignore_index
                    continue

                prev_end = 0
                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    if idx == 0:
                        batch["labels"][i, :start] = self.ignore_index
                    else:
                        batch["labels"][i, prev_end:start] = self.ignore_index
                    prev_end = end

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1]:] = self.ignore_index

        return batch
