import nemo_run as run

from nemo.collections.llm.recipes import llama3_8b

if __name__ == "__main__":
    pretrain = llama3_8b.pretrain_recipe(num_nodes=1, num_gpus_per_node=8, performance_mode=True)

    pretrain.trainer.strategy.context_parallel_size = 1
    pretrain.trainer.log_every_n_steps = 1
    pretrain.data.global_batch_size = 16
    pretrain.data.seq_length = 64
    pretrain.trainer.max_steps = 10

    pretrain.trainer.strategy.fsdp = 'megatron'
    pretrain.trainer.strategy.ddp.average_in_collective = False
    pretrain.trainer.strategy.ddp.use_megatron_fsdp = True
    pretrain.trainer.strategy.save_ckpt_format = 'fsdp_dtensor'
    # pretrain.trainer.strategy.gradient_accumulation_fusion=False

    # # included in the performance mode but not normal mode

    pretrain.trainer.strategy.ddp.grad_reduce_in_fp32 = False
    pretrain.trainer.plugins.grad_reduce_in_fp32 = False
    pretrain.optim.config.use_precision_aware_optimizer = False
    pretrain.optim.config.use_megatron_fsdp = True
    # pretrain.data.seq_length = 4096

    run.run(pretrain)
