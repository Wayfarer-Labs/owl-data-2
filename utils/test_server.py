import zmq
import torch

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://127.0.0.1:5555")

class VideoServerLoader:
    def __init__(self, num_workers=64):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        for i in range(num_workers):
            self.socket.connect(f"tcp://127.0.0.1:{5555 + i}")

    def get_next(self):
        # Receives a Python object sent by the server
        payload = self.socket.recv_pyobj()
        return payload

if __name__ == "__main__":
    import time

    loader = VideoServerLoader()
    num_videos = 1000
    print(f"Testing speed: loading {num_videos} videos...")

    start = time.time()
    for i in range(num_videos):
        data = loader.get_next()
        if (i + 1) % 10 == 0:
            print(f"Loaded {i+1} videos")
    end = time.time()

    elapsed = end - start
    vids_per_sec = num_videos / elapsed if elapsed > 0 else float('inf')
    print(f"Loaded {num_videos} videos in {elapsed:.2f} seconds ({vids_per_sec:.2f} videos/sec)")