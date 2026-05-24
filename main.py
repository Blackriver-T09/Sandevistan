#!/usr/bin/env python3
import sys
from pathlib import Path

from src.config import Config
from src.detector import PersonDetector
from src.renderer import ContourRenderer
from src.video_processor import VideoProcessor
from src.trail_manager import TrailManager


def main(input_video: str, config_path: str = None):
    config = Config(config_path)
    
    detector = PersonDetector(
        model_name=config.model_name,
        conf=config.model_conf,
        iou=config.model_iou,
        device=config.model_device,
        classes=config.model_classes
    )
    
    trail_manager = None
    if config.trail_enabled:
        from src.colormap import ColorMapper
        
        trail_manager = TrailManager(
            sample_interval=config.trail_sample_interval
        )
        print(f"Trail effect enabled: sample every {config.trail_sample_interval} frames")
    
    renderer = ContourRenderer(
        contour_color=config.contour_color,
        contour_thickness=config.contour_thickness,
        fill_color=config.fill_color,
        fill_alpha=config.fill_alpha,
        enable_trails=config.trail_enabled
    )
    
    if config.trail_enabled:
        renderer.color_mapper = ColorMapper(
            alpha=config.trail_alpha,
            saturation=config.trail_saturation,
            hue_step=config.trail_hue_step,
            hue_start=config.trail_hue_start
        )
    
    def process_frame(frame):
        masks = detector.get_masks(frame)
        
        if trail_manager:
            if masks:
                trail_manager.add_trail(frame, masks[0])
            
            trails = trail_manager.get_trails()
            result = renderer.render(frame, masks, trails)
            
            trail_manager.update()
        else:
            result = renderer.render(frame, masks)
        
        return result
    
    processor = VideoProcessor(
        input_path=input_video,
        output_suffix=config.video_output_suffix,
        codec=config.video_codec,
        fps=config.video_fps
    )
    
    output_path = processor.open()
    processor.process(process_frame)
    processor.close()
    
    print(f"\n✓ Video processing complete!")
    print(f"Merging audio...")
    
    if processor.merge_audio(output_path):
        print(f"✓ Audio merged successfully!")
    
    print(f"\n✓ All done!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video> [config_path]")
        print("\nExample:")
        print("  python main.py test.mp4")
        print("  python main.py test.mov config/custom_config.yaml")
        sys.exit(1)
    
    input_video = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(input_video, config_path)
