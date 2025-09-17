import gradio as gr

from owl_debug.overlay_controls import overlay_controls
from owl_debug.download_tar import TarDownloader

import tempfile
import os

def start_gradio_viewer():
    import gradio as gr

    downloader = TarDownloader()

    def download_and_overlay():
        # Download a random eligible tar and extract it
        tmp_dir = downloader.temp_download()
        # Find a video file and a controls file in the extracted directory
        video_file = None
        controls_file = None
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".mp4") or fname.endswith(".mkv") or fname.endswith(".avi"):
                video_file = os.path.join(tmp_dir, fname)
            elif fname.endswith(".csv"):
                controls_file = os.path.join(tmp_dir, fname)
        if not video_file or not controls_file:
            raise RuntimeError("Could not find both video and controls file in the archive.")

        # Create a temporary file for the output video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_video_out = tmpfile.name

        # Run overlay_controls to generate the overlayed video
        overlay_controls(tmp_dir, temp_video_out)

        return temp_video_out

    with gr.Blocks() as demo:
        gr.Markdown("# Controls Overlay Viewer\nDownload a random video and controls, overlay, and view the result.")
        video_output = gr.Video(label="Overlayed Controls Video")
        download_btn = gr.Button("Download Random Video and Overlay Controls")
        download_btn.click(fn=download_and_overlay, inputs=[], outputs=video_output)

    demo.launch(share=True)

if __name__ == "__main__":
    start_gradio_viewer()
