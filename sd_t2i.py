from sd_pipe import StableDiffusionPipeline
from transformers import CLIPTokenizer
from diffusers import PNDMScheduler

pipe = StableDiffusionPipeline(
    vae_decoder_path = "/home/shicheng/workspace/dev_env/PythonCode/StableDiffusion-TPU/models/vae_decoder/vae_decoder_1684x_f16.bmodel",
    te_encoder_path="/home/shicheng/workspace/dev_env/PythonCode/StableDiffusion-TPU/models/text_encoder/text_encoder_1684x_f32.bmodel",
    tokenizer=CLIPTokenizer.from_pretrained("/home/shicheng/workspace/run_env/sophon-demo/sample/StableDiffusionV1_5/models/tokenizer_path"),
    unet_path="/home/shicheng/workspace/dev_env/PythonCode/StableDiffusion-TPU/models/unet/t2i/unet_1684x_f16.bmodel",
    scheduler = PNDMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    skip_prk_steps = True,
    ),
    dev_id=0)

result = pipe(prompt = "a rabbit drinking at the bar", negative_prompt = "worst quality")
result.save("result.png")

