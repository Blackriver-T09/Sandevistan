import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "effect_config.yaml"
        
        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)
    
    @property
    def contour_color(self):
        return tuple(self.data['contour']['color'])
    
    @property
    def contour_thickness(self):
        return self.data['contour']['thickness']
    
    @property
    def fill_color(self):
        return tuple(self.data['fill']['color'])
    
    @property
    def fill_alpha(self):
        return self.data['fill']['alpha']
    
    @property
    def model_name(self):
        return self.data['model']['name']
    
    @property
    def model_conf(self):
        return self.data['model']['conf']
    
    @property
    def model_iou(self):
        return self.data['model']['iou']
    
    @property
    def model_device(self):
        return self.data['model']['device']
    
    @property
    def model_classes(self):
        return self.data['model']['classes']
    
    @property
    def video_output_suffix(self):
        return self.data['video']['output_suffix']
    
    @property
    def video_codec(self):
        return self.data['video']['codec']
    
    @property
    def video_fps(self):
        return self.data['video']['fps']
    
    @property
    def trail_enabled(self):
        return self.data.get('trail', {}).get('enabled', False)
    
    @property
    def trail_sample_interval(self):
        return self.data.get('trail', {}).get('sample_interval', 3)
    
    @property
    def trail_alpha(self):
        return self.data.get('trail', {}).get('alpha', 0.5)
    
    @property
    def trail_saturation(self):
        return self.data.get('trail', {}).get('saturation', 120)
    
    @property
    def trail_hue_step(self):
        return self.data.get('trail', {}).get('hue_step', 5)
