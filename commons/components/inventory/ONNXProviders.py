from enum import Enum

class ONNXProviderModel(Enum):
    
    CUDA = 'CUDAExecutionProvider'
    CPU = 'CPUExecutionProvider'

if __name__ == "__main__":
    
    print(ONNXProviderModel.CUDA)