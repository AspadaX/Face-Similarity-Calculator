from typing import List

import cv2
import torch
from insightface.app import FaceAnalysis

from .inventory.ONNXProviders import ONNXProviderModel


# The `FaceIDRetriever` class is used to retrieve face embeddings from an image using the InsightFace
# model.
class FaceIDRetriever:
    
    def __init__(self, image: str | cv2.typing.MatLike, providers: ONNXProviderModel = ONNXProviderModel.CPU) -> None:
        
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="./models/insightface",
            providers=providers.value
        )
        
        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )
        
        if isinstance(image, str):
            self.image: cv2.typing.MatLike = self.__image_loader(image)
        else: 
            self.image: cv2.typing.MatLike = image
            
        self.faces: list = self.__get_face()
        self.face_embedding: torch.Tensor = self.__get_face_embedding()
    
    def __image_loader(self, image_path: str) -> cv2.typing.MatLike:
        # load a image
        return cv2.imread(image_path)
    
    def __get_face(self) -> list:
        # retrieve faces from the picture
        return self.app.get(self.image)
    
    def __get_face_embedding(self) -> torch.Tensor:
        return torch.from_numpy(self.faces[0].normed_embedding).unsqueeze(0)


if __name__ == "__main__": 
    
    image_path: str = "./IMG_1150.JPG"
    
    face_id_retriever = FaceIDRetriever(providers=ONNXProviderModel.CUDA, image=image_path)
    print(face_id_retriever.face_embedding)