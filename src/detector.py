import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_name: str, conf: float, iou: float, device: str, classes: list):
        print(f"Loading model: {model_name} on {device}...")
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        print("Model loaded successfully!")
    
    def detect(self, frame: np.ndarray):
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False
        )
        
        return results[0]
    
    def get_masks(self, frame: np.ndarray):
        result = self.detect(frame)
        
        if result.masks is None:
            return []
        
        masks = []
        for mask in result.masks.data:
            mask_np = mask.cpu().numpy()
            mask_resized = self._resize_mask(mask_np, frame.shape[:2])
            masks.append(mask_resized)
        
        return masks
    
    def _resize_mask(self, mask: np.ndarray, target_shape: tuple):
        import cv2
        h, w = target_shape
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        return mask_binary
