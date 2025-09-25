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

from nemo.collections.common.data.fallback import FallbackDataset
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import itertools

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec


class DualDataLoader:
    """
    A data loader that alternates between two separate data loaders.
    
    This ensures homogeneous batches - either all transcribe data or all not_transcribe data.
    """
    
    def __init__(self, transcribe_loader, not_transcribe_loader, transcribe_ratio=0.5):
        self.transcribe_loader = transcribe_loader
        self.not_transcribe_loader = not_transcribe_loader
        self.transcribe_ratio = transcribe_ratio
        
        # Calculate alternation pattern
        if transcribe_ratio == 0.0:
            self.pattern = [False]  # Only not_transcribe
        elif transcribe_ratio == 1.0:
            self.pattern = [True]   # Only transcribe
        else:
            # Create alternating pattern based on ratio
            transcribe_count = max(1, int(transcribe_ratio * 10))
            not_transcribe_count = max(1, int((1 - transcribe_ratio) * 10))
            self.pattern = [True] * transcribe_count + [False] * not_transcribe_count
        
        logging.info(f"DualDataLoader created with pattern: {self.pattern}")
    
    def __iter__(self):
        transcribe_iter = iter(self.transcribe_loader)
        not_transcribe_iter = iter(self.not_transcribe_loader)
        
        pattern_cycle = itertools.cycle(self.pattern)
        batch_count = 0
        transcribe_count = 0
        not_transcribe_count = 0
        
        print("="*100)
        print(f"DEBUG: Starting DualDataLoader iteration with pattern: {self.pattern}")
        print(f"DEBUG: Transcribe ratio: {self.transcribe_ratio}")
        print("="*100)
        
        while True:
            try:
                use_transcribe = next(pattern_cycle)
                
                if use_transcribe:
                    try:
                        batch = next(transcribe_iter)
                        transcribe_count += 1
                        if batch_count % 50 == 0:  # Print every 50 batches
                            print(f"DEBUG: Batch {batch_count}: Using transcribe data (transcribe: {transcribe_count}, not_transcribe: {not_transcribe_count})")
                        yield batch
                    except StopIteration:
                        # Restart transcribe loader
                        transcribe_iter = iter(self.transcribe_loader)
                        batch = next(transcribe_iter)
                        transcribe_count += 1
                        yield batch
                else:
                    try:
                        batch = next(not_transcribe_iter)
                        not_transcribe_count += 1
                        if batch_count % 50 == 0:  # Print every 50 batches
                            print(f"DEBUG: Batch {batch_count}: Using not_transcribe data (transcribe: {transcribe_count}, not_transcribe: {not_transcribe_count})")
                        yield batch
                    except StopIteration:
                        # Restart not_transcribe loader
                        not_transcribe_iter = iter(self.not_transcribe_loader)
                        batch = next(not_transcribe_iter)
                        not_transcribe_count += 1
                        yield batch
                
                batch_count += 1
                
            except StopIteration:
                break
    
    def __len__(self):
        return len(self.transcribe_loader) + len(self.not_transcribe_loader)


class HybridSALMTDTDataModule(LightningDataModule):
    """
    A Lightning DataModule specialized for Hybrid SALM-TDT models.
    
    This DataModule handles the separation of speech and non-speech data,
    creating appropriate dataloaders for each type of data.
    
    The typical structure of the YAML config used to initialize this module looks like:
    
    .. code-block:: yaml
    
        data:
          train_ds:
            input_cfg: path/to/input_cfg.yaml
            num_workers: 2
            batch_size: 4
            speech_ratio: 0.5  # Optional: ratio of speech data in batches
            
          validation_ds:
            datasets:
              speech_val:
                cuts_path: path/to/speech_validation.cuts
              non_speech_val:
                cuts_path: path/to/non_speech_validation.cuts
            batch_size: 4
    
    Args:
        cfg: a DictConfig instance, typically corresponding to `data` namespace in YAML configs.
        tokenizer: a tokenizer instance, typically NeMo's AutoTokenizer wrapping HF's AutoTokenizer.
        dataset: a torch.utils.data.Dataset instance, expected to define __getitem__ that accepts
            a lhotse.CutSet. It converts metadata + raw data to a batch of PyTorch tensors.
    """

    def __init__(self, cfg, tokenizer: TokenizerSpec, dataset: torch.utils.data.Dataset, tdt_tokenizer: TokenizerSpec = None) -> None:
        super().__init__()
        self.cfg = cfg
        with open_dict(self.cfg):
            for k in ("validation_ds", "test_ds"):
                if k in self.cfg:
                    getattr(self.cfg, k).force_finite = True
                    # Don't force map dataset for hybrid SALM-TDT as it's designed for iterable datasets
                    # getattr(self.cfg, k).force_map_dataset = True
        self.tokenizer = tokenizer
        self.tdt_tokenizer = tdt_tokenizer
        self.dataset = dataset

    def train_dataloader(self):
        """Create training dataloader with separate transcribe and not_transcribe loaders."""
        if "train_ds" not in self.cfg:
            return None
        
        # Get speech ratio from config, default to 0.5
        speech_ratio = self.cfg.train_ds.get('speech_ratio', 0.5)
        
        # Create separate configs for transcribe and not_transcribe data
        transcribe_cfg = self.cfg.train_ds.copy()
        not_transcribe_cfg = self.cfg.train_ds.copy()
        
        # Modify input_cfg to separate transcribe and not_transcribe datasets
        transcribe_input_cfg = []
        not_transcribe_input_cfg = []
        
        for input_config in self.cfg.train_ds.input_cfg:
            if input_config.get('prompt') == 'transcribe':
                transcribe_input_cfg.append(input_config)
            elif input_config.get('prompt') == 'not_transcribe':
                not_transcribe_input_cfg.append(input_config)
            else:
                # Skip datasets without explicit prompt labels to avoid mixed batches
                logging.warning(f"Skipping dataset without prompt label: {input_config.get('type', 'unknown')}")
                continue
        
        # Handle case when all datasets have the same prompt
        if not transcribe_input_cfg and not_transcribe_input_cfg:
            # All datasets are not_transcribe, duplicate them as transcribe
            logging.warning("All datasets have prompt='not_transcribe'. Duplicating them as transcribe datasets for balanced training.")
            transcribe_input_cfg = not_transcribe_input_cfg.copy()
        elif not not_transcribe_input_cfg and transcribe_input_cfg:
            # All datasets are transcribe, duplicate them as not_transcribe
            logging.warning("All datasets have prompt='transcribe'. Duplicating them as not_transcribe datasets for balanced training.")
            not_transcribe_input_cfg = transcribe_input_cfg.copy()
        elif not transcribe_input_cfg and not not_transcribe_input_cfg:
            # No datasets with explicit prompt labels
            raise ValueError("No datasets with prompt='transcribe' or prompt='not_transcribe' found! All datasets need explicit prompt labels.")
        
        # Ensure we have data for both loaders (after fallback handling)
        if not transcribe_input_cfg:
            raise ValueError("No datasets with prompt='transcribe' found after fallback handling!")
        if not not_transcribe_input_cfg:
            raise ValueError("No datasets with prompt='not_transcribe' found after fallback handling!")
        
        # Update configs
        with open_dict(transcribe_cfg):
            transcribe_cfg.input_cfg = transcribe_input_cfg
        with open_dict(not_transcribe_cfg):
            not_transcribe_cfg.input_cfg = not_transcribe_input_cfg
        
        logging.info(f"Creating transcribe dataloader with {len(transcribe_input_cfg)} datasets")
        logging.info(f"Creating not_transcribe dataloader with {len(not_transcribe_input_cfg)} datasets")
        
        # Check if we used fallback duplication
        original_transcribe_count = len([cfg for cfg in self.cfg.train_ds.input_cfg if cfg.get('prompt') == 'transcribe'])
        original_not_transcribe_count = len([cfg for cfg in self.cfg.train_ds.input_cfg if cfg.get('prompt') == 'not_transcribe'])
        
        if original_transcribe_count == 0 and original_not_transcribe_count > 0:
            logging.info(f"FALLBACK: Duplicated {len(not_transcribe_input_cfg)} not_transcribe datasets as transcribe datasets")
        elif original_not_transcribe_count == 0 and original_transcribe_count > 0:
            logging.info(f"FALLBACK: Duplicated {len(transcribe_input_cfg)} transcribe datasets as not_transcribe datasets")
        
        # Debug: Print dataset configurations with weights
        print("="*100)
        print("DEBUG: Transcribe datasets configuration:")
        for i, cfg in enumerate(transcribe_input_cfg):
            weight = cfg.get('weight', 'Not specified')
            manifest_path = cfg.get('manifest_filepath', 'Unknown')
            print(f"  Dataset {i+1}: weight={weight}, manifest={manifest_path}")
        print("="*100)
        
        print("="*100)
        print("DEBUG: Not-transcribe datasets configuration:")
        for i, cfg in enumerate(not_transcribe_input_cfg):
            weight = cfg.get('weight', 'Not specified')
            manifest_path = cfg.get('manifest_filepath', 'Unknown')
            print(f"  Dataset {i+1}: weight={weight}, manifest={manifest_path}")
        print("="*100)
        
        # Create separate data loaders - each gets its own FallbackDataset instance
        transcribe_loader = get_lhotse_dataloader_from_config(
            config=transcribe_cfg,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=FallbackDataset(self.dataset),  # Instance A - only sees speech data
            tokenizer=self.tokenizer,
            tdt_tokenizer=self.tdt_tokenizer,
        )
        
        not_transcribe_loader = get_lhotse_dataloader_from_config(
            config=not_transcribe_cfg,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=FallbackDataset(self.dataset),  # Instance B - only sees non-speech data
            tokenizer=self.tokenizer,
            tdt_tokenizer=self.tdt_tokenizer,
        )
        
        # Return dual data loader that alternates between the two
        return DualDataLoader(
            transcribe_loader=transcribe_loader,
            not_transcribe_loader=not_transcribe_loader,
            transcribe_ratio=speech_ratio
        )

    def val_dataloader(self):
        """Create validation dataloader with separate speech and non-speech datasets."""
        if "validation_ds" not in self.cfg:
            return None
        cfg = self.cfg.validation_ds
        return self._build_test_dataloader(cfg)

    def test_dataloader(self):
        """Create test dataloader with separate speech and non-speech datasets."""
        if "test_ds" not in self.cfg:
            return None
        cfg = self.cfg.test_ds
        return self._build_test_dataloader(cfg)

    def _build_test_dataloader(self, cfg: DictConfig) -> torch.utils.data.DataLoader | CombinedLoader:
        """Build test/validation dataloader with support for separate speech and non-speech datasets."""
        # Single validation/test dataloader (legacy support)
        if "datasets" not in cfg:
            with open_dict(cfg):
                cfg.force_finite = True
                cfg.force_map_dataset = True
                # Don't force map dataset for hybrid SALM-TDT as it's designed for iterable datasets
                # cfg.force_map_dataset = True
            return get_lhotse_dataloader_from_config(
                config=cfg,
                global_rank=self._get_dp_rank(),
                world_size=self._get_world_size(),
                dataset=self.dataset,
                tokenizer=self.tokenizer,
                tdt_tokenizer=self.tdt_tokenizer,
            )

        # Multiple validation/test dataloaders with speech/non-speech separation
        base_cfg = cfg.copy()
        with open_dict(base_cfg):
            del base_cfg.datasets
        
        dloaders = {}
        for name, item in cfg.datasets.items():
            with open_dict(base_cfg):
                item = OmegaConf.merge(base_cfg, item)
            dloaders[name] = self._build_test_dataloader(item)
        
        return CombinedLoader(dloaders, mode="max_size")

    def _get_dp_rank(self):
        """Get data parallel rank."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.get_coordinate()[0]
            else:
                return torch.distributed.get_rank()  # plain ol' DDP
        else:
            return 0  # 1 GPU

    def _get_world_size(self):
        """Get world size for distributed training."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer, "model")
                and hasattr(self.trainer.model, "device_mesh")
                and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.shape[0]
            else:  # plain ol' DDP
                return torch.distributed.get_world_size()
        else:
            return 1  # 1 GPU
