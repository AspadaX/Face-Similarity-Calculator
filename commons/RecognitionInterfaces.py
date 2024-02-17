import asyncio

from torch.nn.functional import cosine_similarity

from commons.components.FaceIDComponents import FaceIDRetriever
from commons.components.inventory.RecognitionBaseModels import RecognitionInput, RecognitionInputs


class SingleImageInterface:
    
    def __init__(self, parameters: RecognitionInput) -> None:
        self.parameters: RecognitionInput = parameters
    
    async def compute_face_similarity(self) -> float:
        
        face_id_source_image = FaceIDRetriever(
            providers=self.parameters.provider, 
            image=self.parameters.source_image
        )
        
        face_id_target_image = FaceIDRetriever(
            providers=self.parameters.provider,
            image=self.parameters.target_image
        )
        
        return float(
            cosine_similarity(
                face_id_source_image.face_embedding,
                face_id_target_image.face_embedding
            ).item()
        )


class MultiImageInterface:
    
    def __init__(self, parameters: RecognitionInputs) -> None:
        
        # we don't initialize the inherited class here, 
        # as we will need to initialize it separately
        self.parameters = parameters
    
    async def batch_compute_face_similarity(self) -> list:
        tasks: list = [
            SingleImageInterface(item).compute_face_similarity() 
            for item in self.parameters.inputs
        ]
        
        results = asyncio.gather(*tasks, return_exceptions=True)
        
        return results