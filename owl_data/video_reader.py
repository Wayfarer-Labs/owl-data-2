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


if __name__ == "__main__":
    # Search for the first .mp4 file in the "datasets" folder
    mp4_path = None
    for root, dirs, files in os.walk("datasets"):
        for file in files:
            if file.lower().endswith(".mkv"):
                mp4_path = os.path.join(root, file)
                break
        if mp4_path:
            break

    if mp4_path:
        try:
            container = av.open(mp4_path)
            video_stream = container.streams.video[0]
            frame_index = 0
            target_frame = None
            for frame in container.decode(video_stream):
                if frame_index == 99:  # 100th frame (0-based index)
                    frame_array = frame.to_ndarray(format='rgb24')
                    pil_image = Image.fromarray(frame_array)
                    pil_image.save("sample.jpg", "JPEG")
                    print(f"Saved 100th frame from {mp4_path} to sample.jpg")
                    break
                frame_index += 1
            container.close()
            if frame_index < 99:
                print(f"Video {mp4_path} has less than 100 frames.")
        except Exception as e:
            print(f"Error extracting 100th frame: {e}")
    else:
        print("No .mp4 files found in the 'datasets' directory.")
