import torch.nn as nn
import torch

class CA_PreconvModule(nn.Module):
    """
    Custom Cross-Attention process before the conv_in layer of the U-Net.
    """

    def __init__(self, inside_dim=320, nb_heads=8, text_dim=768):
        super(CA_PreconvModule, self).__init__()

        self.nb_heads = nb_heads
        self.CA_Preconv_to_q = nn.Linear(8, inside_dim, bias=False)
        self.CA_Preconv_to_k = nn.Linear(text_dim, inside_dim, bias=False)
        self.CA_Preconv_to_v = nn.Linear(text_dim, inside_dim, bias=False)
        self.CA_Preconv_to_out = nn.Linear(inside_dim, 4, bias=False)

    def forward(self, vae_images, text_embedding):
        
        batch_size, channel, height, width = vae_images.shape
        vae_images = vae_images.reshape(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = vae_images.shape

        query = self.CA_Preconv_to_q(vae_images)
        key = self.CA_Preconv_to_k(text_embedding)
        value = self.CA_Preconv_to_v(text_embedding)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.nb_heads

        query = query.view(batch_size, -1, self.nb_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.nb_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.nb_heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        vae_images = nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        vae_images = nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        vae_images = vae_images.transpose(1, 2).reshape(batch_size, -1, self.nb_heads * head_dim)
        vae_images = vae_images.to(query.dtype)

        vae_images = self.CA_Preconv_to_out(vae_images)
        vae_images = vae_images.transpose(1, 2).reshape(batch_size, 4, height, width)

        return vae_images



def single_noise_pred(sample, embeds, scheduler, unet, timestep, image, guidance_scale, image_guidance_scale, style_image, style_image_guidance_scale, ca_preconv):
    """
    Noise prediction forward pass
    
    Args:
        sample (dict): Dictionary containing the latent samples and timesteps
        embeds (torch.Tensor): Text embeddings for conditioning
        scheduler: Diffusion scheduler
        unet (nn.Module): U-Net model that predicts the noise
        timestep (int): Current timestep index
        image (torch.Tensor): Image conditioning input
        guidance_scale (float): Text guidance scale factor
        image_guidance_scale (float): Image guidance scale factor
        style_image (torch.Tensor): Style image input for style transfer
        style_image_guidance_scale (float): Style image guidance scale factor
        ca_preconv (nn.Module): Cross-attention preconvolution module
        
    Returns:
        torch.Tensor: Predicted noise for the current timestep
    """
    
    # Prepare input latents
    indices = [torch.nonzero(scheduler.timesteps == ts).item() for ts in sample["timesteps"][:, timestep]]
    scaled_latent_model_input = sample["latents"][:, timestep] / ((scheduler.sigmas[indices].view(-1, 1, 1, 1).clone().to(sample["latents"][:, timestep].device)**2 + 1) ** 0.5)
    # Cross-attention
    style_image = torch.cat([style_image, sample["latents"][:, timestep]], 1)
    style_image = ca_preconv(style_image, embeds.chunk(4)[0])
    # Prepare for U-Net
    style_image = torch.cat([style_image, torch.zeros_like(style_image), torch.zeros_like(style_image), style_image], dim=0)
    scaled_latent_model_input = torch.cat([scaled_latent_model_input] * 4)
    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image, style_image], dim=1)
    
    # U-Net forward pass
    noise_pred = unet(
                    scaled_latent_model_input,
                    torch.cat([sample["timesteps"][:, timestep]] * 4),
                    encoder_hidden_states=embeds,
                    added_cond_kwargs=None,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
    # Classifier free guidance
    noise_pred_text, noise_pred_image, noise_pred_uncond, noise_pred_image_style = noise_pred.chunk(4)
    noise_pred = (
        noise_pred_uncond
        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        + style_image_guidance_scale * (noise_pred_image_style - noise_pred_image)
        + guidance_scale * (noise_pred_text - noise_pred_image_style)
    )

    return noise_pred
