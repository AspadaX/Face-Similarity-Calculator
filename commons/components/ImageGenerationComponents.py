# The `IPAdapterGenerationEngine` class is an image generation engine that uses a stable diffusion
# pipeline, a VAE model, and an IP adapter model to generate images based on prompts and face ID
# embeddings.
import os

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from PIL import Image


# The `ImageGenerationEngine` class is responsible for initiating the 
# necessary components used when generating images using a stable diffusion
# model and a variational autoencoder.
class ImageGenerationEngine:
    
    def __init__(
        self, 
        base_model_path: str, 
        vae_model_path: str, 
        clip_model_path: str,
        device: str = 'cpu'
    ) -> None:
        
        self.vae_model_path: str = vae_model_path
        self.clip_model_path: str = clip_model_path
        self.device: str = device
        
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        self.noise_scheduler: DDIMScheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        
        self.vae: AutoencoderKL = AutoencoderKL.from_single_file(
            pretrained_model_link_or_path=self.vae_model_path,
            device=self.device
        )
        
        # load the CLIP model that is necessary for Stable Diffusion
        self.clip_text_model: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.clip_model_path
        )
        self.clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            self.clip_model_path
        )
        
        self.stable_diffusion_model: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=base_model_path,
            torch_dtype=torch.float16,
            text_encoder=self.clip_text_model,
            tokenizer=self.clip_tokenizer,
            local_files_only=True,
            load_safety_checker=False
        ).to(self.device)

# The `IPAdapterGenerationEngine` class is a subclass of `ImageGenerationEngine` that generates images
# using an IPAdapterFaceID model.
class IPAdapterGenerationEngine(ImageGenerationEngine):
    
    def __init__(
        self, 
        base_model_path: str, 
        vae_model_path: str, 
        clip_model_path: str,
        ip_adapter_checkpoint_path: str, 
        device: str = 'cpu'
    ) -> None:
        super().__init__(
            base_model_path=base_model_path, 
            vae_model_path=vae_model_path, 
            clip_model_path=clip_model_path, 
            device=device,
        )
        
        self.ip_adapter_checkpoint_path: str = ip_adapter_checkpoint_path
        
        self.ip_adapter_model: IPAdapterFaceID = IPAdapterFaceID(
            sd_pipe=self.stable_diffusion_model,
            ip_ckpt=self.ip_adapter_checkpoint_path,
            device=self.device
        )
        
    def generate_image(
        self, 
        prompt: str, 
        face_id_embeddings: torch.Tensor, 
        negative_prompt: str = None, 
        number_of_samples: int = 4, 
        width: int = 512, 
        height: int = 512, 
        number_of_inference_steps: int = 30, 
        seed: int = None
    ) -> Image:
        
        """
        The `generate_image` function takes in a prompt, face ID embeddings, and other optional
        parameters, and generates an image with a stable diffusion model through an IP adapter model.
        
        :param prompt: The prompt is a string that represents the desired image generation task. It
        provides instructions or a description of what the generated image should look like
        :type prompt: str
        :param face_id_embeddings: The `face_id_embeddings` parameter is a tensor containing the
        embeddings of the face images. These embeddings are typically obtained using a face recognition
        model and represent the unique features of each face. The embeddings are used to guide the
        generation of new images that resemble the input faces
        :type face_id_embeddings: torch.Tensor
        :param negative_prompt: The `negative_prompt` parameter is an optional string that represents a
        negative prompt. It is used to guide the generation of the image in a direction opposite to the
        given prompt. If not provided, it defaults to an empty string
        :type negative_prompt: str
        :param number_of_samples: The parameter `number_of_samples` determines the number of images to
        generate, defaults to 4
        :type number_of_samples: int (optional)
        :param width: The `width` parameter specifies the width of the generated image in pixels,
        defaults to 512
        :type width: int (optional)
        :param height: The `height` parameter specifies the height of the generated image in pixels,
        defaults to 512
        :type height: int (optional)
        :param number_of_inference_steps: The parameter "number_of_inference_steps" determines the
        number of steps or iterations the model will take to generate the image. Increasing the number
        of inference steps can potentially improve the quality and detail of the generated image, but it
        will also increase the time taken for the generation process, defaults to 30
        :type number_of_inference_steps: int (optional)
        :param seed: The `seed` parameter is an optional parameter that allows you to set a specific
        seed value for random number generation. If no seed is provided, a random seed will be generated
        using `torch.randint` function
        :type seed: int
        :return: an output image of type `Image`.
        """
        
        if seed is None:
            seed = torch.randint(low=0, high=1000000000, size=(1,)).item()
        
        if negative_prompt is None:
            negative_prompt = ""
        
        image = self.ip_adapter_model.generate(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            face_id_embeddings=face_id_embeddings, 
            number_of_samples=number_of_samples, 
            width=width, 
            height=height, 
            number_of_inference_steps=number_of_inference_steps, 
            seed=seed
        )
        
        output_image: Image = Image.open(image)
        
        return output_image