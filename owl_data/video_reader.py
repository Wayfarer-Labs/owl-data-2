import os
import av
import numpy as np
from PIL import Image

def extract_frames(video_path, output_path, frame_skip):
    """
    Extract frames from a video file and save them as JPEG images.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Directory where frames will be saved
        frame_skip (int): Number of frames to skip between saved frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load video with pyav
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Error: Could not open video file {video_path}: {e}")
        return
    
    # Get the video stream
    video_stream = container.streams.video[0]
    
    try:
        frame_count = 0
        saved_frame_index = 0
        
        for frame in container.decode(video_stream):
            # Save frame if it's at the right interval
            if frame_count % (frame_skip + 1) == 0:
                # Convert frame to numpy array (RGB format)
                frame_array = frame.to_ndarray(format='rgb24')
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_array)
                
                # Create filename with zero-padded index
                filename = f"{saved_frame_index:06d}.jpg"
                filepath = os.path.join(output_path, filename)
                
                # Save as JPEG
                pil_image.save(filepath, "JPEG")
                
                saved_frame_index += 1
            
            frame_count += 1
            
    except Exception as e:
        print(f"Error processing frames from {video_path}: {e}")
    finally:
        # Close the container
        container.close()
    
    print(f"Extracted {saved_frame_index} frames from {video_path} to {output_path}")

class VideoReader:
    def __init__(self, video_path, frame_skip=1):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.container = av.open(video_path)
        self.video_stream = self.container.streams.video[0]
        self._frame_iter = self.container.decode(self.video_stream)
        self._frame_count = 0

    def get_avg_fps(self):
        pass

    def __iter__(self):
        self._frame_iter = self.container.decode(self.video_stream)
        self._frame_count = 0
        return self

    def __next__(self):
        """
        Iterates over frames, returned array is HWC, RGB, uint8 np array
        """
        for frame in self._frame_iter:
            if self._frame_count % (self.frame_skip + 1) == 0:
                frame_array = frame.to_ndarray(format='rgb24')
                self._frame_count += 1
                return frame_array
            self._frame_count += 1
        raise StopIteration

    def __del__(self):
        try:
            self.container.close()
        except Exception:
            pass

if __name__ == "__main__":
    vid_path = "/mnt/data/shahbuland/video-proc-2/datasets/cod-yt/99-youtube video #KxYXtXBtpEE.mp4"

    reader = VideoReader(vid_path)
    for frame in reader:
        print(frame.shape)
        break