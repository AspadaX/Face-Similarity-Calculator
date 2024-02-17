from pydantic import BaseModel

class IPAdapterGenerationGenerationInterfaceTypeTemplate(BaseModel):
    stable_diffusion_model: str
    vae: str
    clip: str
    ip_adapter: str
    device: str