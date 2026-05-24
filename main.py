#!/usr/bin/env python3
import sys
from pathlib import Path

from src.config import Config
from src.detector import PersonDetector
from src.renderer import ContourRenderer
from src.video_processor import VideoProcessor


def main(input_video: str, config_path: str = None):
    config = Config(config_path)
    
    detector = PersonDetector(
        model_name=config.model_name,
        conf=config.model_conf,
        iou=config.model_iou,
        device=config.model_device,
        classes=config.model_classes
    )
    
    renderer = ContourRenderer(
        contour_color=config.contour_color,
        contour_thickness=config.contour_thickness,
        fill_color=config.fill_color,
        fill_alpha=config.fill_alpha
    )
    
    def process_frame(frame):
        masks = detector.get_masks(frame)
        result = renderer.render(frame, masks)
        return result
    
    with VideoProcessor(
        input_path=input_video,
        output_suffix=config.video_output_suffix,
        codec=config.video_codec,
        fps=config.video_fps
    ) as processor:
        output_path = processor.open()
        processor.process(process_frame)
    
    print(f"\n✓ Processing complete!")
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
