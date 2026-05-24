import cv2
import numpy as np
from .colormap import ColorMapper


class ContourRenderer:
    def __init__(self, contour_color: tuple, contour_thickness: int, 
                 fill_color: tuple, fill_alpha: float, enable_trails: bool = False):
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        self.fill_color = fill_color
        self.fill_alpha = fill_alpha
        self.enable_trails = enable_trails
        self.color_mapper = ColorMapper()
    
    def render(self, frame: np.ndarray, masks: list, trails: list = None) -> np.ndarray:
        overlay = frame.copy()
        
        if self.enable_trails and trails:
            current_mask = masks[0] if masks else None
            overlay = self._render_trails(overlay, trails, current_mask)
        
        return overlay
    
    def _render_trails(self, frame: np.ndarray, trails: list, 
                       current_mask: np.ndarray = None) -> np.ndarray:
        result = frame.copy()
        
        if not trails:
            return result
        
        for trail in trails:
            color = self.color_mapper.get_color_by_index(trail.color_index)
            alpha = self.color_mapper.get_alpha()
            
            trail_mask = trail.mask.copy()
            
            if current_mask is not None:
                trail_mask = trail_mask & (~current_mask.astype(bool)).astype(np.uint8)
            
            result = self._render_trail_mask(result, trail.frame_region, 
                                            trail_mask, color, alpha)
        
        return result
    
    def _render_trail_mask(self, frame: np.ndarray, trail_frame: np.ndarray,
                           mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
        result = frame.copy()
        
        color_overlay = np.zeros_like(trail_frame)
        color_overlay[:] = color
        
        tinted_frame = cv2.addWeighted(trail_frame, 1.0 - alpha, color_overlay, alpha, 0)
        
        mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
        result[mask_3ch] = tinted_frame[mask_3ch]
        
        return result
    
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
        h, w = frame.shape[:2]
        
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
        
        self._draw_edge_extensions(frame, mask, h, w)
        
        return frame
    
    def _draw_edge_extensions(self, frame: np.ndarray, mask: np.ndarray, h: int, w: int):
        edge_margin = 10
        
        bottom_region = mask[-edge_margin:, :]
        if np.any(bottom_region > 0):
            cols_with_mask = np.where(np.any(bottom_region > 0, axis=0))[0]
            if len(cols_with_mask) > 0:
                for col in cols_with_mask:
                    rows_with_mask = np.where(bottom_region[:, col] > 0)[0]
                    if len(rows_with_mask) > 0:
                        last_row = h - edge_margin + rows_with_mask[-1]
                        cv2.line(frame, (col, last_row), (col, h-1), 
                                self.contour_color, self.contour_thickness)
        
        top_region = mask[:edge_margin, :]
        if np.any(top_region > 0):
            cols_with_mask = np.where(np.any(top_region > 0, axis=0))[0]
            if len(cols_with_mask) > 0:
                for col in cols_with_mask:
                    rows_with_mask = np.where(top_region[:, col] > 0)[0]
                    if len(rows_with_mask) > 0:
                        first_row = rows_with_mask[0]
                        cv2.line(frame, (col, first_row), (col, 0), 
                                self.contour_color, self.contour_thickness)
        
        left_region = mask[:, :edge_margin]
        if np.any(left_region > 0):
            rows_with_mask = np.where(np.any(left_region > 0, axis=1))[0]
            if len(rows_with_mask) > 0:
                for row in rows_with_mask:
                    cols_with_mask = np.where(left_region[row, :] > 0)[0]
                    if len(cols_with_mask) > 0:
                        first_col = cols_with_mask[0]
                        cv2.line(frame, (first_col, row), (0, row), 
                                self.contour_color, self.contour_thickness)
        
        right_region = mask[:, -edge_margin:]
        if np.any(right_region > 0):
            rows_with_mask = np.where(np.any(right_region > 0, axis=1))[0]
            if len(rows_with_mask) > 0:
                for row in rows_with_mask:
                    cols_with_mask = np.where(right_region[row, :] > 0)[0]
                    if len(cols_with_mask) > 0:
                        last_col = w - edge_margin + cols_with_mask[-1]
                        cv2.line(frame, (last_col, row), (w-1, row), 
                                self.contour_color, self.contour_thickness)
