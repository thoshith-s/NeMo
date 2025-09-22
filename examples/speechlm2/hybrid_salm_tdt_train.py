# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

# Add the NeMo path to import the hybrid model
sys.path.append('/lustre/fsw/portfolios/convai/users/lgrigoryan/NeMo')

from nemo.collections.speechlm2.data.hybrid_salm_tdt_datamodule import HybridSALMTDTDataModule
from nemo.collections.speechlm2.data.hybrid_salm_tdt_dataset import HybridSALMTDTDataset
from nemo.collections.speechlm2.models.hybrid_salm_tdt import HybridSALMTDT
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="/lustre/fsw/portfolios/convai/users/lgrigoryan/nemotron-omni/hybrid_salm_tdt/conf/model_confs", config_name="canary-qwen-tdt-2.5b")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup trainer
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    # Initialize hybrid model
    with trainer.init_module():
        model = HybridSALMTDT(OmegaConf.to_container(cfg.model, resolve=True))
    
    print(f"Hybrid model initialized successfully!")
    print(f"TDT weight: {model.tdt_weight}")
    print(f"TDT tokenizer type: {model.tdt_tokenizer_type}")
    print(f"TDT vocab size: {len(model.tdt_tokenizer.tokenizer.get_vocab())}")
    print(f"TDT decoder: {type(model.tdt_decoder).__name__}")
    print(f"TDT joint: {type(model.tdt_joint).__name__}")
    print(f"TDT loss: {type(model.tdt_loss).__name__}")
    print(f"Pretrained ASR model: {model.cfg.pretrained_asr}")
    print(f"Assert identical encoders: {model.perception.encoder == model.asr_model.encoder}")

    speech_ratio = cfg.data.train_ds.get('speech_ratio', 0.5)
    speech_dataset = HybridSALMTDTDataset(
        salm_tokenizer=model.tokenizer, 
        tdt_tokenizer=model.tdt_tokenizer,
    )
    datamodule = HybridSALMTDTDataModule(
        cfg.data, 
        tokenizer=model.tokenizer, 
        dataset=speech_dataset,
        tdt_tokenizer=model.tdt_tokenizer,
    )

    print(f"Hybrid dataset and datamodule setup completed")
    print(f"Speech ratio: {speech_ratio}")
    print(f"DataModule will create separate loaders:")
    print(f"  - Transcribe loader: speech data → TDT + SALM heads")
    print(f"  - Not_transcribe loader: non-speech data → SALM head only")
    print(f"  - Alternating pattern ensures homogeneous batches")

    # Start training
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    
    train()
