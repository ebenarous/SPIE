# Adapted from https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py

from typing import Optional, Union
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import EulerAncestralDiscreteScheduler

def eulerancestral_step_with_logprob(
    self: EulerAncestralDiscreteScheduler,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
    prev_sample: Optional[torch.FloatTensor] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor.

    """
    assert isinstance(self, EulerAncestralDiscreteScheduler)

    if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if not self.is_scale_input_called:
        logger.warning(
            "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
            "See `StableDiffusionPipeline` for a usage example."
        )

    # Training with shuffled timesteps requires to reinitialize the step index   
    if timestep.ndim == 0:
        self._init_step_index(timestep)
        sigma = self.sigmas[self.step_index]
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
    else: # Adjust for batch training with different timesteps for each element in the batch
        sigma = []
        sigma_from = []
        sigma_to = []
        for ts in timestep:
            self._init_step_index(ts)
            sigma.append(self.sigmas[self.step_index])
            sigma_from.append(self.sigmas[self.step_index])
            sigma_to.append(self.sigmas[self.step_index + 1])
        sigma = torch.tensor(sigma).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(sample.device)
        sigma_from = torch.tensor(sigma_from).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(sample.device)
        sigma_to = torch.tensor(sigma_to).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(sample.device)
    
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if self.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == "v_prediction":
        # * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    elif self.config.prediction_type == "sample":
        raise NotImplementedError("prediction_type not implemented yet: sample")
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma

    dt = sigma_down - sigma

    prev_sample_mean = sample + derivative * dt
    
    if prev_sample is None:
        device = model_output.device
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        prev_sample = prev_sample_mean + noise * sigma_up

    # Calculate the log probability of prev_sample
    variance = sigma_up ** 2 + 1e-8 # avoid div by 0
    log_prob = -0.5 * (((prev_sample - prev_sample_mean) ** 2) / variance
                        + torch.log(2 * torch.pi * variance))
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(model_output.dtype), log_prob
