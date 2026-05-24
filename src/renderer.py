import cv2
import numpy as np


class ContourRenderer:
    def __init__(self, contour_color: tuple, contour_thickness: int, 
                 fill_color: tuple, fill_alpha: float):
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        self.fill_color = fill_color
        self.fill_alpha = fill_alpha
    
    def render(self, frame: np.ndarray, masks: list) -> np.ndarray:
        if not masks:
            return frame
        
        overlay = frame.copy()
        
        for mask in masks:
            overlay = self._render_single_mask(overlay, mask)
        
        return overlay
    
    def _render_single_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = frame.copy()
        
        result = self._apply_fill(result, mask)
        result = self._draw_contour(result, mask)
        
        return result
    
    def _apply_fill(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = self.fill_color
        
        result = cv2.addWeighted(frame, 1.0, colored_mask, self.fill_alpha, 0)
        return result
    
    def _draw_contour(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        cv2.drawContours(
            frame, 
            contours, 
            -1, 
            self.contour_color, 
            self.contour_thickness
        )
        
        return frame
