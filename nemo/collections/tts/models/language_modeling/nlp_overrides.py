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

import itertools
import os
import shutil
import tempfile
from typing import Optional, Union

import torch
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import ckpt_to_dir, uninject_model_parallel_rank

try:
    from megatron.core import parallel_state

    from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from modelopt.torch.opt.plugins import restore_sharded_modelopt_state, save_sharded_modelopt_state

    HAVE_MODELOPT = True

except Exception:
    HAVE_MODELOPT = False


class NLPSaveRestoreConnector(SaveRestoreConnector):
    """Custom connector to support saving and restoring states."""

    def __init__(self) -> None:
        if not HAVE_MEGATRON_CORE:
            logging.warning(
                "megatron-core was not found. Please see the NeMo README for installation instructions: "
                "https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__()

    def save_to(self, model, save_path: str):
        """Save model to save path."""
        app_state = AppState()

        # Check if using distributed checkpointing
        if model.cfg.get("fsdp", False):
            dist_ckpt = False
        else:
            dist_ckpt = hasattr(model, 'sharded_state_dict') and model.sharded_state_dict() is not None

        dist_ckpt_dir = None

        if (app_state.model_parallel_size is not None and app_state.model_parallel_size > 1) or dist_ckpt:

            dir_name = os.path.dirname(save_path)

            # dist ckpt calls save on every rank
            if dist_ckpt:
                # model weights is a directory
                dist_ckpt_dir = ckpt_to_dir(os.path.join(dir_name, self.model_weights_ckpt))
                # dist checkpoint needs torch.distributed to save the checkpoint
                if not parallel_state.is_initialized():

                    def dummy():
                        return

                    if model.trainer.strategy.launcher is not None:
                        model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
                    model.trainer.strategy.setup_environment()
                sharded_state_dict = model.sharded_state_dict()
                checkpoint_io = DistributedCheckpointIO.from_config(model.cfg, async_save=False)
                checkpoint_io.save_checkpoint(sharded_state_dict, dist_ckpt_dir)

                if HAVE_MODELOPT and hasattr(model, "get_model_module_list"):
                    while isinstance(checkpoint_io, _WrappingCheckpointIO):
                        checkpoint_io = checkpoint_io.checkpoint_io
                    save_sharded_modelopt_state(
                        model.get_model_module_list(),
                        dist_ckpt_dir,
                        checkpoint_io.save_sharded_strategy,
                        prefix="model.",
                    )

            else:

                # first we save the weights for each model parallel rank
                if app_state.data_parallel_rank == 0:
                    if app_state.pipeline_model_parallel_size == 1:
                        mp_model_weights = os.path.join(
                            dir_name, f'mp_rank_{app_state.tensor_model_parallel_rank:02d}_' + self.model_weights_ckpt
                        )
                    else:
                        mp_model_weights = os.path.join(
                            dir_name,
                            f'tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_'
                            f'{app_state.pipeline_model_parallel_rank:03d}_' + self.model_weights_ckpt,
                        )

                    self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # create nemo file from folder with all mp_ranks checkpoints
            if dist_ckpt:
                should_move_data = is_global_rank_zero()
            else:
                should_move_data = (
                    app_state.pipeline_model_parallel_rank == 0
                    and app_state.tensor_model_parallel_rank == 0
                    and app_state.data_parallel_rank == 0
                )

            if should_move_data:
                with tempfile.TemporaryDirectory() as tmpdir:
                    if dist_ckpt:
                        shutil.move(str(dist_ckpt_dir), tmpdir)
                    elif app_state.pipeline_model_parallel_size == 1:
                        # move weights to the tmpdir
                        for tp_rank in range(app_state.tensor_model_parallel_size):
                            os.makedirs(os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}'))
                            mp_model_weights = os.path.join(
                                dir_name, f'mp_rank_{tp_rank:02d}_' + self.model_weights_ckpt
                            )
                            shutil.move(
                                mp_model_weights,
                                os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}', self.model_weights_ckpt),
                            )
                    else:
                        # move weights to the tmpdir
                        for tp_rank, pp_rank in itertools.product(
                            range(app_state.tensor_model_parallel_size),
                            range(app_state.pipeline_model_parallel_size),
                        ):
                            os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}'))
                            mp_model_weights = os.path.join(
                                dir_name, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}_' + self.model_weights_ckpt
                            )
                            shutil.move(
                                mp_model_weights,
                                os.path.join(
                                    tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}', self.model_weights_ckpt
                                ),
                            )

                    # create config and artifacts in tmpdir
                    config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                    model.to_config_file(path2yaml_file=config_yaml)
                    if hasattr(model, 'artifacts') and model.artifacts is not None:
                        self._handle_artifacts(model, nemo_file_folder=tmpdir)
                        self._update_artifact_paths(model, path2yaml_file=config_yaml)

                    # create tar file
                    if self.pack_nemo_file:
                        self._make_nemo_file_from_folder(save_path, tmpdir)
                    else:
                        # Get the folder path from the save_path and move all values inside the tmpdir to the folder
                        folder_path = os.path.dirname(save_path)

                        for file in os.listdir(tmpdir):
                            shutil.move(os.path.join(tmpdir, file), folder_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        else:
            return super().save_to(model, save_path)

    def modify_state_dict(self, conf, state_dict):
        """Remap keys in state dict."""
        if conf.get('megatron_legacy', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('bert_model.language_model', 'bert_model.model.language_model')
                new_key = new_key.replace('transformer', 'encoder')
                new_key = new_key.replace('.attention.', '.self_attention.')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        if conf.get('megatron_amp_O2', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('model.', 'model.module.', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace(
                'word_embeddings.adapter_layer.mm_linear_adapter.linear',
                'word_embeddings.adapter_layer.mm_projector_adapter.mm_projector',
                1,
            )
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict

        # compatibility for inductor in inference
        if not conf.get('inductor', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('._orig_mod', '', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        # Modify state key for Dreambooth inference
        if (
            conf.get('target')
            == 'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm.MegatronLatentDiffusion'
        ):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('unet', 'model.diffusion_model')
                new_key = new_key.replace('vae', 'first_stage_model')
                new_key = new_key.replace('text_encoder', 'cond_stage_model')
                new_key = new_key.replace('.noise_scheduler', '')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        loaded_keys = state_dict.keys()
        if 'model.model.diffusion_model.input_blocks.1.0.in_layers.2.weight' in loaded_keys:
            new_state_dict = {}

            # GroupNormOpt fuses activation function to one layer, thus the indexing of weights are
            # shifted for following
            def should_process(key):
                base_str = "model.model.diffusion_model."
                blocks = ["input_blocks", "middle_block", "output_blocks"]
                for block in blocks:
                    for layer_type in ["in_layers", "out_layers"]:
                        for index in [2, 3]:  # The layers index.
                            for param in ["weight", "bias"]:
                                if block == 'middle_block':
                                    for num in [0, 2]:
                                        template = f"{base_str}{block}.{num}.{layer_type}.{index}.{param}"
                                        if key == template:
                                            return True
                                else:
                                    for num in range(12):  # 12 blocks, adjust as needed.
                                        template = f"{base_str}{block}.{num}.0.{layer_type}.{index}.{param}"
                                        if key == template:
                                            return True
                return False

            for key_ in state_dict.keys():
                if key_ == "model.cond_stage_model.transformer.text_model.embeddings.position_ids":
                    continue
                if should_process(key_):
                    s = key_.split('.')
                    idx = int(s[-2])
                    new_key_ = ".".join(s[:-2] + [str(int(idx - 1))] + [s[-1]])
                    new_state_dict[new_key_] = state_dict[key_]
                else:
                    new_state_dict[key_] = state_dict[key_]
            state_dict = new_state_dict

        if conf.get('unet_config') and conf.get('unet_config').get('use_te_fp8') == False:
            # Mapping potential fp8 ckpt to fp16 model
            # remove _extra_state in fp8 if there is.
            new_state_dict = {}
            for key in state_dict.keys():
                if 'extra_state' in key:
                    continue

                # LayerNormLinear
                # norm_to_q.layer_norm_{weight|bias} -> norm.{weight|bias}
                # norm_to_q.weight -> to_q.weight
                new_key = key.replace('norm_to_q.layer_norm_', 'norm.')
                new_key = new_key.replace('norm_to_q.weight', 'to_q.weight')

                # LayerNormMLP
                # ff.net.layer_norm_{weight|bias} -> ff.net.0.{weight|bias}
                # ff.net.fc1_{weight|bias} -> ff.net.1.proj.{weight|bias}
                # ff.net.fc2_{weight|bias} -> ff.net.3.{weight|bias}
                new_key = new_key.replace('ff.net.layer_norm_', 'ff.net.0.')
                new_key = new_key.replace('ff.net.fc1_', 'ff.net.1.proj.')
                new_key = new_key.replace('ff.net.fc2_', 'ff.net.3.')

                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        return state_dict

    def _load_state_dict_from_disk(self, model_weights, map_location=None):
        # if model_weights with the extension removed is a directory, we assume it is a distributed checkpoint
        # we need to defer loading the state dict so we return None
        uninject_model_weights = uninject_model_parallel_rank(model_weights)

        # legacy model_weights will have mp rank injected
        if os.path.isfile(model_weights):
            return super()._load_state_dict_from_disk(model_weights, map_location)

        # dist checkpoint will be a dir
        elif os.path.isdir(os.path.splitext(uninject_model_weights)[0]):
            return None
        else:
            raise ValueError(f'Expected {model_weights} to be a file or directory.')

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
        validate_access_integrity: bool = True,
        replace_sharded_tensor_key: Optional[str] = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.nlp.models.TextClassification.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.nlp.models.TextClassification)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """

        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls,
            restore_path,
            override_config_path,
            map_location,
            strict,
            return_config,
            trainer,
            validate_access_integrity,
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        conf, instance, state_dict = loaded_params

        # if we're using dist checkpointing then state_dict will be None
        if state_dict is None:
            # dist checkpointing needs torch.distributed to load the checkpoint
            if not parallel_state.is_initialized():

                def dummy():
                    return

                if trainer.strategy.launcher is not None:
                    trainer.strategy.launcher.launch(dummy, trainer=trainer)
                trainer.strategy.setup_environment()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Check if self.model_extracted_dir is set, and is a valid path
                if self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir):
                    # Log that NeMo will use the provided `model_extracted_dir`
                    logging.info(
                        f"Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`."
                    )

                    # Override `tmpdir` above with the pre-extracted `model_extracted_dir`
                    tmpdir = self.model_extracted_dir

                else:
                    # Extract the nemo file into the temporary directory
                    filter_fn = None
                    if return_config:
                        filter_fn = lambda name: '.yaml' in name
                    members = self._filtered_tar_info(restore_path, filter_fn=filter_fn)
                    self._unpack_nemo_file(
                        path2file=restore_path,
                        out_folder=tmpdir,
                        members=members,
                    )
                # remove model weights extension
                tmp_model_weights_ckpt = os.path.join(tmpdir, self.model_weights_ckpt)
                tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
                assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'

                if HAVE_MODELOPT and hasattr(instance, "get_model_module_list"):
                    restore_sharded_modelopt_state(
                        instance.get_model_module_list(), tmp_model_weights_dir, prefix="model."
                    )

                checkpoint = {}
                sharded_state_dict = instance.sharded_state_dict()
                checkpoint['state_dict'] = sharded_state_dict
                if replace_sharded_tensor_key:
                    for v in checkpoint["state_dict"].values():
                        if hasattr(v, "key"):
                            v.key = v.key.replace("model", replace_sharded_tensor_key)

                checkpoint_io = DistributedCheckpointIO.from_config(conf)
                checkpoint = checkpoint_io.load_checkpoint(
                    tmp_model_weights_dir,
                    sharded_state_dict=checkpoint,
                    strict=strict,
                    validate_access_integrity=validate_access_integrity,
                )
                instance.on_load_checkpoint(checkpoint)
                if hasattr(instance, 'setup_transformer_engine_tp_groups'):
                    instance.setup_transformer_engine_tp_groups()

        else:
            state_dict = self.modify_state_dict(conf, state_dict)
            super().load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance
