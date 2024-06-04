import inspect
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .sd_engine import EngineOV

def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    timesteps = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusionPipeline:
    def __init__(
        self,
        # vae_encoder_path,
        vae_decoder_path,
        te_encoder_path,
        tokenizer,
        unet_path,
        scheduler,
        dev_id = 0,
        safety_checker = None,
        feature_extracter = None,
        image_encoder = None,
        requires_safety_checker = None):

        # module config parameter
        self.device = dev_id
        self.scheduler = scheduler
        self.unet_config_in_channels = 4
        self.unet_config_sample_size = 64
        self.vae_scale_factor = 8
        self.tokenizer = tokenizer

        # load model
        self.unet = EngineOV(unet_path, device_id = dev_id)
        self.text_encoder = EngineOV(te_encoder_path, device_id = dev_id)
        # self.vae_encoder = EngineOV(vae_encoder_path, device_id = dev_id)
        self.vae_decoder = EngineOV(vae_decoder_path, device_id = dev_id)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        # callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def encode_prompt(
        self,
        prompt,
        # device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        # lora_scale = None,
        clip_skip = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # if lora_scale is not None and isinstance(self, LoraLoaderMixin):
        #     self._lora_scale = lora_scale

        #     # dynamically adjust the LoRA scale
        #     if not USE_PEFT_BACKEND:
        #         adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
        #     else:
        #         scale_lora_layers(self.text_encoder, lora_scale)

        assert isinstance(prompt, str)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            #     text_input_ids, untruncated_ids
            # ):
            #     removed_text = self.tokenizer.batch_decode(
            #         untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            #     )
            #     logger.warning(
            #         "The following part of your input was truncated because CLIP can only handle sequences up to"
            #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #     )

            # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            #     attention_mask = text_inputs.attention_mask.to(device)
            # else:
            #     attention_mask = None

            if clip_skip is None:
                # prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = self.text_encoder({"tokens": np.array(text_input_ids)})
                prompt_embeds = prompt_embeds[0]
            else:
                pass
                # prompt_embeds = self.text_encoder(
                #     text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                # )
                # # Access the `hidden_states` first, that contains a tuple of
                # # all the hidden states from the encoder layers. Then index into
                # # the tuple to access the hidden states from the desired layer.
                # prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # # We also need to apply the final LayerNorm here to not mess with the
                # # representations. The `last_hidden_states` that we typically use for
                # # obtaining the final prompt representations passes through the LayerNorm
                # # layer.
                # prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        # if self.text_encoder is not None:
        #     prompt_embeds_dtype = self.text_encoder.dtype
        # elif self.unet is not None:
        #     prompt_embeds_dtype = self.unet.dtype
        # else:
        #     prompt_embeds_dtype = prompt_embeds.dtype

        # prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, axis = 1)
        prompt_embeds = prompt_embeds.reshape(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None

            # negative_prompt_embeds = self.text_encoder(
                # uncond_input.input_ids.to(device),
                # attention_mask=attention_mask,
            # )
            uncond_input_ids = uncond_input.input_ids
            negative_prompt_embeds = self.text_encoder({"tokens": np.array(uncond_input_ids)})

            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            # negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, axis = 1)
            negative_prompt_embeds = negative_prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)

        # if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
        #     # Retrieve the original scale by scaling back the LoRA layers
        #     unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def __call__(
        self,
        prompt = None,
        height = None,
        width = None,
        num_inference_steps = 20,
        timesteps = None,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        ip_adapter_image = None,
        output_type = "pil",
        return_dict = True,
        cross_attention_kwargs = None,
        guidance_rescale = 0.0,
        clip_skip = None,
        callback_on_step_end = None,
        # callback_on_step_end_tensor_inputs = ["latents"],
        callback_steps = 1,
        **kwargs,
        ):
        callback = kwargs.pop("callback", None)

        # 0. Default height and width to unet
        height = height or self.unet_config_sample_size * self.vae_scale_factor
        width = width or self.unet_config_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            # callback_on_step_end_tensor_inputs,
        )
        # self._guidance_scale = guidance_scale
        # self._guidance_rescale = guidance_rescale
        # self._clip_skip = clip_skip
        # self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        assert isinstance(prompt, str)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1

        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device

        # 3. Encode input prompt
        # lora_scale = None
        # lora_scale = (
        #     self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        # )
        self.do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            # device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # lora_scale=lora_scale,
            # clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis = 0)

        # if ip_adapter_image is not None:
        #     image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
        #     if self.do_classifier_free_guidance:
        #         image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)

        # 5. Prepare latent variables
        # num_channels_latents = self.unet.config.in_channels
        num_channels_latents = self.unet_config_in_channels

        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        latent_shape = (batch_size * num_images_per_prompt, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = np.random.randn(*latent_shape).astype(prompt_embeds.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        # added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 6.2 Optionally get Guidance Scale Embedding
        # timestep_cond = None
        # if self.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        #     timestep_cond = self.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                newt = np.array([t])
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if self.do_classifier_free_guidance else latents
                # TODO: scale_model_input depends on scheduler type, it could be deprecate
                # >>>>>>>>>>>> this code block could be deprecated.
                latent_model_input = torch.from_numpy(latent_model_input)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.numpy()
                # <<<<<<<<<<<<

                # predict the noise residual
                noise_pred = self.unet(
                    {'latent.1':latent_model_input,
                    't.1':newt,
                    'prompt_embeds.1': prompt_embeds,}
                    # timestep_cond=timestep_cond,
                    # cross_attention_kwargs=self.cross_attention_kwargs,
                    # added_cond_kwargs=added_cond_kwargs,
                    # return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    # noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                temp_noise_pred = torch.from_numpy(noise_pred)
                temp_latents = torch.from_numpy(latents)
                latents = self.scheduler.step(temp_noise_pred, t, temp_latents, return_dict=False)[0]
                latents = latents.numpy()

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # if not output_type == "latent":
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
        #         0
        #     ]
        #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
            # return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        # self.vae.config.scaleing_factor is 0.18125, varies among different vae
        image = self.vae_decoder({"x.1": latents / 0.18125})[0]
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        pil_img = Image.fromarray(image)

        return pil_img