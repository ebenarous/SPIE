import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ############ General ############
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 0
    # top-level logging directory for checkpoint saving.
    config.logdir = "./logs"
    # directory for saving generated samples (for inference-only script)
    config.sample_savedir = "./generated_samples"
    # directory for loading input images to be edited / trained on
    config.inputimg_dir = "your_input_data_path"
    # directory for loading style images to be edited / trained on
    config.styleimg_dir = "your_style_data_path"
    # name of input element to edit
    config.input_name = "your_input_data_name"
    # name of style element to reproduce
    config.style_name = "your_style_data_name"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.num_epochs = 20
    # number of epochs between saving model checkpoints.
    config.save_freq = 1
    # number of checkpoints to keep before overwriting old ones. (0 means keep all checkpoints).
    config.num_checkpoint_limit = 0
    # checkpoint path
    config.checkpoint_path = "./checkpoints"
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "no"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # generate samples or resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.load_checkpoint = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    # LoRA Rank
    config.lora_rank = 8
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False

    ############ Pretrained Model ############
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "timbrooks/instruct-pix2pix"
    # revision of the model to load.
    pretrained.revision = "main"
    # Scheduler to be used
    pretrained.scheduler = "euler_ancestral" #"ddim"

    ############ Sampling ############
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 100
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 7.5
    # classifier-free guidance for input image
    sample.image_guidance_scale = 1.5
    # classifier-free guidance for style image
    sample.style_image_guidance_scale = 3.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 4
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 2
    # resize the loaded base image to a smaller resolution to train quicker (None or int)
    sample.data_resize = 256
    # Number of edits to generate for one input sample (binary->2)
    sample.nb_gen = 2

    ############ Training ############
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 8
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    train.activation_checkpointing = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # coefficient of the KL divergence
    train.beta = 0.1
    # The coefficient constraining the probability ratio. Equivalent to restricting the Q-values within a certain range.
    train.eps = 0.1
    # weight of semantic score relative to structural score
    train.score_alpha = 1
    # pixel reconstruction regularizer
    train.rec_lambda = 1
    # Edit entire frame
    train.whole_frame = False

    return config
