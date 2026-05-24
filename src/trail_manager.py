import numpy as np


class TrailFrame:
    def __init__(self, frame_region: np.ndarray, mask: np.ndarray, 
                 frame_index: int, color_index: int):
        self.frame_region = frame_region
        self.mask = mask
        self.frame_index = frame_index
        self.color_index = color_index


class TrailManager:
    def __init__(self, sample_interval: int = 3):
        self.sample_interval = sample_interval
        self.trails = []
        self.frame_counter = 0
        self.color_counter = 0
    
    def should_sample(self) -> bool:
        return self.frame_counter % self.sample_interval == 0
    
    def add_trail(self, frame: np.ndarray, mask: np.ndarray):
        if self.should_sample() and mask is not None:
            frame_region = frame.copy()
            trail = TrailFrame(frame_region, mask.copy(), 
                             self.frame_counter, self.color_counter)
            self.trails.append(trail)
            self.color_counter += 1
    
    def update(self):
        self.frame_counter += 1
    
    def get_trails(self):
        return self.trails
    
    def clear(self):
        self.trails.clear()
        self.frame_counter = 0
        self.color_counter = 0
