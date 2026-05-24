import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, input_path: str, output_suffix: str = "_processed", 
                 codec: str = "mp4v", fps: float = None):
        self.input_path = Path(input_path)
        self.output_suffix = output_suffix
        self.codec = codec
        self.fps = fps
        
        self.cap = None
        self.writer = None
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.input_fps = 0
    
    def open(self):
        self.cap = cv2.VideoCapture(str(self.input_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        output_fps = self.fps if self.fps else self.input_fps
        output_path = self._get_output_path()
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (self.frame_width, self.frame_height)
        )
        
        print(f"Input: {self.input_path}")
        print(f"Output: {output_path}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {output_fps:.2f}")
        print(f"Total frames: {self.total_frames}")
        
        return output_path
    
    def process(self, frame_callback):
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        
        pbar = tqdm(total=self.total_frames, desc="Processing", unit="frame")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = frame_callback(frame)
            
            self.writer.write(processed_frame)
            pbar.update(1)
        
        pbar.close()
    
    def close(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
    
    def merge_audio(self, video_path: Path):
        temp_path = video_path.parent / f"{video_path.stem}_temp{video_path.suffix}"
        video_path.rename(temp_path)
        
        cmd = [
            'ffmpeg',
            '-i', str(temp_path),
            '-i', str(self.input_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            '-y',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            temp_path.unlink()
            return True
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to merge audio: {e}")
            temp_path.rename(video_path)
            return False
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Audio will not be included.")
            temp_path.rename(video_path)
            return False
    
    def _get_output_path(self):
        stem = self.input_path.stem
        suffix = self.input_path.suffix
        output_name = f"{stem}{self.output_suffix}{suffix}"
        return self.input_path.parent / output_name
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
