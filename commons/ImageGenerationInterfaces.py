import os

from PIL import Image
import torch

from .components.inventory.ImageGenerationTypeTemplates import IPAdapterGenerationGenerationInterfaceTypeTemplate
from .components.inventory.ONNXProviders import ONNXProviderModel
from .components.ImageGenerationComponents import IPAdapterGenerationEngine
from .components.FaceIDComponents import FaceIDRetriever

class IPAdapterGenerationGenerationInterface:
    
    def __init__(self, params: IPAdapterGenerationGenerationInterfaceTypeTemplate) -> None:
        
        self.base_path: str = "models"
        self.stable_diffusion_model_directory: str = "stable_diffusion"
        self.vae_model_directory: str = "vae"
        self.clip_model_directory: str = "tokenizer"
        self.ip_adapter_checkpoint_directory: str = "ip_adapter"
        
        self.base_model_path: str = os.path.join(
            self.base_path, 
            self.stable_diffusion_model_directory, 
            params.stable_diffusion_model
        )
        
        self.vae_model_path: str = os.path.join(
            self.base_path, 
            self.vae_model_directory, 
            params.vae
        )
        
        self.clip_model_path: str = os.path.join(
            self.base_path, 
            self.clip_model_directory, 
            params.clip
        )
        
        self.ip_adapter_checkpoint_path: str = os.path.join(
            self.base_path, 
            self.ip_adapter_checkpoint_directory, 
            params.ip_adapter
        )
        
        self.device: str = params.device
    
    def generate_image(
        self, 
        image: str, 
        prompt: str, 
        negative_prompt: str = None, 
        number_of_samples: int = 4, 
        width: int = 512, 
        height: int = 512, 
        number_of_inference_steps: int = 30, 
        seed: int = None
    ) -> Image:
        
        if self.device == "cuda":
            face_id_embeddings: torch.Tensor = FaceIDRetriever(
                image=image,
                providers=ONNXProviderModel.CUDA
            ).face_embedding
        else:
            face_id_embeddings: torch.Tensor = FaceIDRetriever(
                image=image
            ).face_embedding
        
        image = IPAdapterGenerationEngine(
            base_model_path=self.base_model_path, 
            vae_model_path=self.vae_model_path, 
            clip_model_path=self.clip_model_path,
            ip_adapter_checkpoint_path=self.ip_adapter_checkpoint_path, 
            device=self.device
        ).generate_image(
            prompt=prompt, 
            face_id_embeddings=face_id_embeddings, 
            negative_prompt=negative_prompt, 
            number_of_samples=number_of_samples, 
            width=width, 
            height=height, 
            number_of_inference_steps=number_of_inference_steps, 
            seed=seed
        )
        
        return image