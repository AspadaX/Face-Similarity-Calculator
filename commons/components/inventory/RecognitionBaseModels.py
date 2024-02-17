from typing import List

import cv2
from pydantic import BaseModel

from .ONNXProviders import ONNXProviderModel


class RecognitionInput(BaseModel):
    
    source_image: str | cv2.typing.MatLike
    target_image: str | cv2.typing.MatLike
    provider: ONNXProviderModel
    
    class Config:
        arbitrary_types_allowed = True


class RecognitionInputs(BaseModel):
    
    inputs: List[RecognitionInput]
    
    class Config:
        arbitrary_types_allowed = True