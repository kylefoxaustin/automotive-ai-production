import torch

class ONNXExporter:
    def __init__(self, model):
        self.model = model
    
    def export_onnx(self, path):
        print(f"Exporting to {path}")
        return path
    
    def export_tensorrt(self, path):
        print(f"Exporting TensorRT to {path}")
        return path
