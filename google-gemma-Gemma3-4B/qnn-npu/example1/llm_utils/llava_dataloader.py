#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""  utility method to build and pre-process Llava dataset """
import os
import torch
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
from PIL import Image as PILImage
from transformers.feature_extraction_utils import BatchFeature

def _convert_one_conversation(conversation: list[dict[str, str]], image_dir=None) -> dict[str, str | list[dict]]:
        """Convert the given conversation to LLaVA style.

        Examples:

            >>> conversation = {"from": "human", "value": "<image>What are the colors of the bus in the image?"}
            >>> LlavaConversationProcessor._convert(conversation)
            {
                'role': 'user',
                'content': [{'type': 'image'}, {'type': 'text', 'text': 'What are the colors of the bus in the image?'}]
            }
            >>> conversation = {"from": "gpt", "value": "The bus in the image is white and red."}
            >>> _convert(conversation)
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'The bus in the image is white and red.'}]
            }
        """
        who = conversation.get("from")
        match who:
            case "human":
                role = "user"
            case "gpt":
                role = "assistant"
            case _:
                raise ValueError(f"Unknown role: {who}")

        text = conversation.get("value")

        if "<image>" in text:
            has_image = True
            text = text.replace("<image>", "")
        else:
            has_image = False

        return {
            "role": role,
            "content": (
                [{"type": "image", "image": image_dir}, {"type": "text", "text": text}] if has_image else [{"type": "text", "text": text}]
            ),
        }

def get_llava_dataset(tokenzier, processor, data_files, dataset_path, cache_dir, prompt_test=False):
    def _map(examples):
        if not prompt_test:  # For calibration/optimization
            examples['message'] = [_convert_one_conversation(conversation=conversation, image_dir=os.path.join(dataset_path, examples["image"])) 
                                for conversation in examples['conversations']]
        else:  # For prompt testing
            examples['question'] = examples['conversations'][0]['value']   # set 1st question as prompt
            examples['annotation'] = examples['conversations'][1]['value']   # set gpt response to 1st question as annotation

        return examples

    def _load_image_and_tokenize(example):
        if not prompt_test:  # For calibration/optimization
            inputs = processor.apply_chat_template(
                example['message'], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        else:  # For prompt testing
            inputs = {}
            inputs['question'] = example['question']
            inputs['annotation'] = example['annotation']
            inputs['image_file'] = [os.path.join(dataset_path, example["image"][0])]

        return inputs

    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split='train')
    dataset = dataset.map(_map)
    return dataset.with_transform(_load_image_and_tokenize)


def add_chat_template_llava(question):
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}"}
            ]
        }
    ]

    return message

