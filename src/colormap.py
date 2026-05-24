import numpy as np
import cv2


class ColorMapper:
    def __init__(self, alpha: float = 0.5, saturation: int = 120, 
                 hue_step: int = 5, hue_start: int = 0):
        self.alpha = alpha
        self.saturation = saturation
        self.hue_step = hue_step
        self.hue_start = hue_start
    
    def get_color_by_index(self, color_index: int) -> tuple:
        hue = (self.hue_start + color_index * self.hue_step) % 180
        
        hsv_color = np.uint8([[[hue, self.saturation, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        return tuple(int(c) for c in bgr_color)
    
    def get_alpha(self) -> float:
        return self.alpha
