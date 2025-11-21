# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

sys.path.append(os.path.dirname(__file__))

import random
from pathlib import Path

import numpy as np
import torch
from config import ModelConfig
from datasets import load_dataset
from olive.data.registry import Registry
from transformers import Sam2Processor


class BaseDataLoader:
    def __init__(self, total):
        self.data = []
        self.total = total

    def __getitem__(self, idx):
        if idx >= len(self.data) or idx >= self.total:
            raise StopIteration
        # print(f"Process data {idx}")
        return self.data[idx]

    def load(self, file):
        self.data.append({key: torch.from_numpy(value) for key, value in np.load(file).items()})

    def finish_load(self):
        if len(self.data) > self.total:
            self.data = random.sample(self.data, self.total)


class VeEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        ve_generate_quant_data(total)
        self.data_files = [
            os.path.join(ModelConfig.data_dir, f.name)
            for f in os.scandir(ModelConfig.data_dir)
            if "images.npz" in f.name
        ]
        self.data_files.sort()
        for f in self.data_files:
            self.load(f)
        self.finish_load()


class MdDecoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total, point_p, mask_p):
        super().__init__(total)
        md_generate_quant_data(total, point_p, mask_p)
        self.data_files = [
            os.path.join(ModelConfig.data_dir, f.name)
            for f in os.scandir(ModelConfig.data_dir)
            if "points.npz" in f.name
        ]
        self.data_files.sort()
        for f in self.data_files:
            self.load(f)
        self.finish_load()


class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


def ve_inputs(batch_size, torch_dtype):
    return {
        ModelConfig.ve_input_name: torch.rand(
            (batch_size, ModelConfig.ve_channel_size, ModelConfig.ve_sample_size, ModelConfig.ve_sample_size),
            dtype=torch_dtype,
        )
    }


def md_inputs(batch_size, torch_dtype):
    return {
        input_name: torch.rand((batch_size, *input_shape), dtype=torch_dtype)
        for input_name, input_shape in zip(ModelConfig.md_input_names, ModelConfig.ms_input_shapes)
    }


def ve_conversion_inputs(model=None):
    return tuple(ve_inputs(1, torch.float32).values())


def md_conversion_inputs(model=None):
    return tuple(md_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def ve_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(ve_inputs, batch_size, torch.float32)


@Registry.register_dataloader()
def ve_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return VeEncoderGeneratedDataLoader(data_num)


@Registry.register_dataloader()
def md_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(md_inputs, batch_size, torch.float32)


@Registry.register_dataloader()
def md_quantize_data_loader(dataset, data_num, point_p, mask_p, *args, **kwargs):
    return MdDecoderGeneratedDataLoader(data_num, point_p, mask_p)

    
def ve_generate_quant_data(num_samples):
    p = Path(ModelConfig.data_dir)
    if p.is_dir() and (len([f for f in p.glob("*images.npz")]) >= num_samples):
        return

    processor = Sam2Processor.from_pretrained(ModelConfig.model_name)
    dataset = load_dataset("nielsr/coco-panoptic-val2017")
    dataset = dataset["train"]
    os.makedirs(ModelConfig.data_dir, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        image = sample["image"]
        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].detach().cpu().numpy()
        np.savez(f"{ModelConfig.data_dir}/input_{i}_images.npz", input=pixel_values)


def get_inputs(sample, point_p = 0.0, mask_p = 0.0):
    inputs = {}
    segments_info = sample['segments_info']
    segment_info = np.random.choice(segments_info)
    box = segment_info['bbox']
    p1 = [box[0], box[1]]
    p2 = [p1[0] + box[2], p1[1] + box[3]]
    p = np.mean([p1, p2], axis = 0)
    
    if np.random.random() < point_p:
        inputs['point_coords'] = np.concatenate([[p], np.zeros((4, 2))])[None, :]
        inputs['point_labels'] = np.concatenate([[1], -np.ones(4)])[None, :]
    else:
        inputs['point_coords'] = np.concatenate([[p1, p2], np.zeros((3, 2))])[None, :]
        inputs['point_labels'] = np.concatenate([[2, 3], -np.ones(3)])[None, :]

    if np.random.random() < mask_p:
        pil_mask = sample['label']
        w, h = pil_mask.size
        masks = np.array(pil_mask.resize((256, 256)))
        mask_point = masks[int(p[1]*256/h), int(p[0]*256/w)]
        mask = (masks == mask_point).all(axis = -1)
        inputs['mask_input'] = mask[None, None, :]
        inputs['has_mask_input'] = np.array([1])
    else:
        inputs['mask_input'] = np.zeros((1, 1, 256, 256))
        inputs['has_mask_input'] = np.array([0])
    return inputs


def md_generate_quant_data(num_samples, point_p, mask_p):
    p = Path(ModelConfig.data_dir)
    if p.is_dir() and (len([f for f in p.glob("*points.npz")]) >= num_samples):
        return
    from hydra import initialize
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2
    from generate_model import model_weights_url, checkpoint, model_config_url, model_cfg
    from generate_model import download_file, SAM2Encoder

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    processor = Sam2Processor.from_pretrained(ModelConfig.model_name)
    dataset = load_dataset("nielsr/coco-panoptic-val2017")
    dataset = dataset["train"]
    os.makedirs(ModelConfig.data_dir, exist_ok=True)
    
    download_file(model_weights_url, checkpoint)
    download_file(model_config_url, model_cfg)
    
    GlobalHydra.instance().clear()
    initialize(config_path="./", job_name="sam2_inference", version_base=None)
    sam2_model = build_sam2(model_cfg, checkpoint, device="cpu")
    
    encoder = SAM2Encoder(sam2_model).to(device)
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        image = sample['image']
        inputs = get_inputs(sample, point_p, mask_p)
        process_inputs = processor(image, input_points = [inputs['point_coords']], input_labels = [inputs['point_labels']], return_tensors="pt")
        image_embed, high_feat_1, high_feat2 = encoder(input = process_inputs['pixel_values'].to(device))
        quant_inputs = {}
        quant_inputs['image_embeddings'] = image_embed.detach().cpu().numpy().astype(np.float32)
        quant_inputs['high_res_features1'] = high_feat_1.detach().cpu().numpy().astype(np.float32)
        quant_inputs['high_res_features2'] = high_feat2.detach().cpu().numpy().astype(np.float32)
        quant_inputs['point_coords'] = process_inputs['input_points'].detach().cpu().numpy()[0].astype(np.float32)
        quant_inputs['point_labels'] = process_inputs['input_labels'].detach().cpu().numpy()[0].astype(np.float32)
        quant_inputs['mask_input'] = inputs['mask_input'].astype(np.float32)
        quant_inputs['has_mask_input'] = inputs['has_mask_input'].astype(np.float32)

        np.savez(f"{ModelConfig.data_dir}/input_{i}_points.npz", **quant_inputs)