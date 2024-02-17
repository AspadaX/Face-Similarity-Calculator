import logging

import gradio as gr
import pydantic
import cv2

from commons.RecognitionInterfaces import SingleImageInterface, RecognitionInput
from commons.components.inventory.ONNXProviders import ONNXProviderModel


async def compare_faces(
    source_image: cv2.typing.MatLike, 
    target_image: cv2.typing.MatLike
) -> float:
    """
    we use `cv2.typing.MatLike` here for type hint, however,
    the `FaceIDComponents` supports both a path string and a
    `cv2.typing.MatLike`
    """
    
    try:
        
        return await SingleImageInterface(
            parameters=RecognitionInput(
                source_image=source_image,
                target_image=target_image,
                provider=ONNXProviderModel.CPU
            )
        ).compute_face_similarity()
    
    except pydantic.ValidationError as e:
        logging.error(e)
        
        return 0.0

demo = gr.Interface(
    fn=compare_faces,
    inputs=[
        gr.Image(type="numpy"),
        gr.Image(type="numpy")
    ],
    outputs="text",
)

if __name__ == "__main__":
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )