import cv2
import numpy as np
import pandas as pd
import os

from owl_debug.process_controls import process_controls

def draw_compass(frame, x_center, y_center, x_arrow, y_arrow, radius, color_arrow, color_compass):
    """
    Draw a compass at (x_center, y_center) with a radius of radius.
    Compass is empty circle of color color_compass
    Arrow is drawn inside from (x_center, y_center) to (x_center + x_arrow, y_center + y_arrow) with color color_arrow
    """
    # Draw compass circle
    cv2.circle(frame, (int(x_center), int(y_center)), radius, color_compass, 2)
    
    # Calculate arrow end point
    end_x = int(x_center + x_arrow)
    end_y = int(y_center + y_arrow)
    
    # Draw main arrow line
    cv2.line(frame, (int(x_center), int(y_center)), (end_x, end_y), color_arrow, 2)
    
    # Calculate arrow head
    arrow_length = np.sqrt(x_arrow**2 + y_arrow**2)
    if arrow_length > 0:
        # Normalize arrow direction
        norm_x = x_arrow / arrow_length
        norm_y = y_arrow / arrow_length
        
        # Arrow head parameters
        head_length = min(15, arrow_length * 0.3)
        head_angle = 0.5  # radians
        
        # Calculate arrow head points
        head_x1 = end_x - head_length * (norm_x * np.cos(head_angle) + norm_y * np.sin(head_angle))
        head_y1 = end_y - head_length * (norm_y * np.cos(head_angle) - norm_x * np.sin(head_angle))
        
        head_x2 = end_x - head_length * (norm_x * np.cos(head_angle) - norm_y * np.sin(head_angle))
        head_y2 = end_y - head_length * (norm_y * np.cos(head_angle) + norm_x * np.sin(head_angle))
        
        # Draw arrow head lines
        cv2.line(frame, (end_x, end_y), (int(head_x1), int(head_y1)), color_arrow, 2)
        cv2.line(frame, (end_x, end_y), (int(head_x2), int(head_y2)), color_arrow, 2)
    
    return frame

def draw_buttons(frame, x_tl : int, y_tl : int, size : int, font_size : int, button_gap : int,labels : list[str], states : list[bool], pressed_color):
    """
    Draw boxes to represent buttons being pressed.
    Boxes are red when false, pressed_color when true (i.e. when key/button is pressed)

    Starts drawing boxes at (x_tl,y_tl) all squares with size (size, size) and horizontal gap button_gap between the end of one box and the start of the next box.
    Labels are drawn above the boxes (above their center)
    """
    for i in range(len(labels)):
        x = x_tl + i * (size + button_gap)
        y = y_tl
        cv2.rectangle(frame, (x, y), (x + size, y + size), pressed_color if states[i] else (0, 0, 255), -1)
        cv2.putText(frame, labels[i], (x + size // 2, y + size // 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    return frame

def overlay_controls(data_path, video_out_path, skip_frames = 0, max_frames = None):
    """
    Overlays controls on top of a video

    :param data_path: Path to data directory containing an mp4 and csv
    :param video_out_path: Path to the output video with controls overlayed
    :param skip_frames: Number of frames to skip from start of video (i.e. start later)
    :param max_frames: Maximum number of frames to process
    """

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    mp4_files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]
    
    if not csv_files:
        raise ValueError(f"No CSV file found in {data_path}")
    if not mp4_files:
        raise ValueError(f"No MP4 file found in {data_path}")
    
    controls_in_path = os.path.join(data_path, csv_files[0])
    video_in_path = os.path.join(data_path, mp4_files[0])
    
    controls = process_controls(controls_in_path)
    video = cv2.VideoCapture(video_in_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    # Set up initial state tracking
    key_states = {}
    mouse_states = {1: False, 2: False, 3: False}  # Left, Middle, Right mouse buttons
    
    # Determine all unique keys from the controls data
    unique_keys = set()

    for index, row in controls.iterrows():
        frame_num = row['frame_number']
        event_type = row['event_type']
        event_args = row['event_args']
        if event_type in ['KEYDOWN', 'KEYUP']:
            unique_keys.add(event_args)
    
    # Initialize key states
    for key in unique_keys:
        key_states[key] = False
    
    # Skip frames if needed
    for _ in range(skip_frames):
        ret, frame = video.read()
        if not ret:
            break
    
    frame_idx = skip_frames
    control_idx = 0
    processed_frames = 0
    
    # Mouse position tracking
    mouse_x, mouse_y = 0, 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if max_frames and processed_frames >= max_frames:
            break
        
        # Update control states for this frame
        while control_idx < len(controls) and controls.iloc[control_idx]['frame_number'] <= frame_idx:
            row = controls.iloc[control_idx]
            frame_num = row['frame_number']
            event_type = row['event_type']
            event_args = row['event_args']
            
            if event_type == 'KEYDOWN':
                key_states[event_args] = True
            elif event_type == 'KEYUP':
                key_states[event_args] = False
            elif event_type == 'MOUSEDOWN':
                mouse_states[event_args] = True
            elif event_type == 'MOUSEUP':
                mouse_states[event_args] = False
            elif event_type == 'MOUSE_MOVE':
                mouse_x, mouse_y = event_args[0], event_args[1]
            
            control_idx += 1
        
        # Draw compass for mouse movement (top left, green arrow)
        compass_center_x, compass_center_y = 50, 50
        compass_radius = 30
        cv2.circle(frame, (compass_center_x, compass_center_y), compass_radius, (255, 255, 255), 2)
        
        # Scale mouse movement and draw arrow
        scale_factor = 5
        arrow_end_x = mouse_x * scale_factor
        arrow_end_y = mouse_y * scale_factor
        
        frame = draw_compass(frame, compass_center_x, compass_center_y, arrow_end_x, arrow_end_y, compass_radius, (0, 255, 0), (255, 255, 255))
        
        # Draw keyboard buttons (green when pressed)
        keyboard_labels = [str(key) for key in sorted(unique_keys)]
        keyboard_states = [key_states[int(label)] for label in keyboard_labels]
        
        button_size = 30
        button_gap = 5
        font_size = 0.4
        keyboard_y = height - 120
        keyboard_x = 10
        
        frame = draw_buttons(frame, keyboard_x, keyboard_y, button_size, font_size, button_gap, 
                           keyboard_labels, keyboard_states, (0, 255, 0))
        
        # Draw mouse buttons (blue when pressed)
        mouse_labels = ['L', 'M', 'R']  # Left, Middle, Right
        mouse_button_states = [mouse_states[1], mouse_states[2], mouse_states[3]]
        
        mouse_y = height - 80
        mouse_x = 10
        
        frame = draw_buttons(frame, mouse_x, mouse_y, button_size, font_size, button_gap,
                           mouse_labels, mouse_button_states, (255, 0, 0))
        
        video_writer.write(frame)
        frame_idx += 1
        processed_frames += 1
    
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_path = "overwatch"
    video_out_path = "controls_overlay.mp4"
    overlay_controls(data_path, video_out_path)