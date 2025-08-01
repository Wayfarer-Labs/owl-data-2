import os
import av
import argparse
import ray
import math
from multiprocessing import cpu_count
import zmq
import random

from owl_data.video_reader import VideoReader
import numpy as np
import cv2

"""
Server object loads many videos in a separate process into some queue that another
process can pull from/exhaust.
"""

@ray.remote
class VideoWorker:
    def __init__(
        self,
        root_dir, 
        rank, world_size,
        suffix = '.mp4',
        frame_skip = 1,
        n_frames = 100,
        known_fps = 30,
        queue_max = 2000,
        resize_to = 256,
        sequential = False
    ):
        self.root_dir = root_dir
        self.suffix = suffix
        self.frame_skip = frame_skip
        self.n_frames = n_frames
        self.known_fps = known_fps
        self.queue_max = queue_max
        # Allow resize_to to be a tuple; if not, make it a tuple (square)
        if isinstance(resize_to, tuple):
            self.resize_to = resize_to
        else:
            self.resize_to = (resize_to, resize_to)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        port = 5555 + rank  # Each worker gets its own port
        self.socket.bind(f"tcp://127.0.0.1:{port}")
        self.socket.setsockopt(zmq.SNDHWM, queue_max)
        self.worker_id = rank
        self.sequential = sequential
        self.world_size = world_size

        self.video_paths = self.get_all_video_paths()
        if self.sequential:
            self.video_paths = iter(self.video_paths)

    def get_all_video_paths(self):
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(self.suffix):
                    video_paths.append(os.path.join(dirpath, fname))
        if self.sequential:
            # Deterministically split the list among workers
            video_paths.sort()  # Ensure consistent order
            total = len(video_paths)
            per_worker = total // self.world_size
            remainder = total % self.world_size
            start = self.rank * per_worker + min(self.rank, remainder)
            end = start + per_worker + (1 if self.rank < remainder else 0)
            return video_paths[start:end]
        else:
            return video_paths

    def run(self):
        effective_fps = self.known_fps // self.frame_skip # i.e. 30fps, fs=2, 15fps
        while True:
            if self.sequential:
                try:
                    vid_path = next(self.video_paths)
                except StopIteration:
                    # If we've reached the end of the list, break
                    break
            else:
                vid_path = random.choice(self.video_paths)
            reader = VideoReader(vid_path, self.frame_skip)
            per_vid_cnt = 0

            # Skip random number of frames [0, self.n_frames]
            start_skip = random.randint(0, self.n_frames) if not self.sequential else 0

            frame_buffer = []

            import cv2  # Ensure cv2 is imported at the top of the file if not already

            for i, frame in enumerate(reader):
                if i < start_skip:
                    continue
                frame_buffer.append(frame)
                if len(frame_buffer) >= self.n_frames:
                    # Resize all frames in the buffer before stacking
                    resized_buffer = [
                        cv2.resize(f, self.resize_to, interpolation=cv2.INTER_LINEAR)
                        for f in frame_buffer
                    ]

                    # Get start and end for buffer
                    end_frame = i * self.frame_skip
                    start_frame = end_frame - self.n_frames * self.frame_skip

                    start_ts = start_frame / effective_fps
                    end_ts = end_frame / effective_fps

                    frames = np.stack(resized_buffer) # [nhwc] uint8 [0,255] rgb
                    metadata = {
                        'start_frame' : start_frame,
                        'end_frame' : end_frame,
                        'start_ts' : start_ts,
                        'end_ts' : end_ts,
                        'vid_path' : vid_path,
                        'vid_name' : os.path.basename(vid_path),
                        'vid_dir' : os.path.dirname(vid_path),
                        'idx_in_vid' : per_vid_cnt,
                    }
                    payload = {
                        'frames' : frames,
                        'metadata' : metadata,
                    }
                    print("Sent a payload")
                    self.socket.send_pyobj(payload)
                    frame_buffer = []
                    per_vid_cnt += 1
                    
        print(f"Worker {self.worker_id} finished with no errors")

def main():
    parser = argparse.ArgumentParser(description='Video Loader Server')
    parser.add_argument('--root_dir', type = str, required = True, help = 'Root directory full of MP4s to draw from')
    parser.add_argument('--num_workers', type = int, default = 32, help = 'Number of workers to spawn')
    parser.add_argument('--queue_max', type = int, default=100000, help = 'Max number of videos we store in queue')
    parser.add_argument('--frame_skip', type = int, default=1, help = 'Number of frames to skip between saved frames')
    parser.add_argument('--n_frames', type = int, default=16, help = 'How many frames per data instance?')
    parser.add_argument('--known_fps', type = int, default = 30, help = 'If we know the FPS, we can use it to calculate the number of frames to skip')
    parser.add_argument('--suffix', type = str, default = '.mp4', help = 'Suffix of videos to load')
    parser.add_argument('--resize_to', type = int, nargs='+', default = 256, help = 'Square size or tuple (h w) to resize videos to')
    parser.add_argument('--sequential', action = 'store_true', help = 'Load all videos sequentially, as opposed to randomly')
    args = parser.parse_args()

    # If resize_to is a list of length 1, make it an int; if length 2, make it a tuple
    if isinstance(args.resize_to, list):
        if len(args.resize_to) == 1:
            resize_to = args.resize_to[0]
        elif len(args.resize_to) == 2:
            resize_to = tuple(args.resize_to)
        else:
            raise ValueError("resize_to must be an int or a tuple of length 2")
    else:
        resize_to = args.resize_to

    workers = [VideoWorker.remote(args.root_dir, i, args.num_workers, args.suffix, args.frame_skip, args.n_frames, args.known_fps, args.queue_max, resize_to, args.sequential) for i in range(args.num_workers)]
    ray.get([worker.run.remote() for worker in workers])

if __name__ == "__main__":
    main()