import pandas as pd
import ast
import numpy as np

def process_controls(csv_path, fps = 60):
    df = pd.read_csv(csv_path)
    
    # Find start event and get its timestamp
    start_idx = df[df['event_type'] == 'START'].index
    if len(start_idx) == 0:
        raise ValueError("No START event found in CSV")
    start_timestamp = df.loc[start_idx[0], 'timestamp']
    
    # Find end event if it exists
    end_idx = df[df['event_type'] == 'END'].index
    if len(end_idx) > 0:
        # Remove everything from END event onwards
        df = df.loc[:end_idx[0]-1]
    
    # Remove everything up to and including START event
    df = df.loc[start_idx[0]+1:]
    
    # Offset timestamps from start
    df['timestamp'] = df['timestamp'] - start_timestamp
    
    # Convert timestamps to frame numbers
    df['frame_number'] = (df['timestamp'] * fps).astype(int)
    df = df.drop('timestamp', axis=1)
    
    # Process keyboard and mouse button events
    processed_rows = []
    
    for idx, row in df.iterrows():
        if row['event_type'] in ['KEYBOARD', 'MOUSE_BUTTON']:
            # Parse event_args
            args_str = row['event_args'].replace('false', 'False').replace('true', 'True')
            args = ast.literal_eval(args_str)
            
            key_or_button = args[0]
            is_down = args[1]
            
            if row['event_type'] == 'KEYBOARD':
                new_event_type = 'KEYDOWN' if is_down else 'KEYUP'
            else:  # MOUSE_BUTTON
                new_event_type = 'MOUSEDOWN' if is_down else 'MOUSEUP'
            
            processed_rows.append({
                'frame_number': row['frame_number'],
                'event_type': new_event_type,
                'event_args': key_or_button
            })
        elif row['event_type'] == 'MOUSE_MOVE':
            # Parse mouse move args
            args = ast.literal_eval(row['event_args'])
            processed_rows.append({
                'frame_number': row['frame_number'],
                'event_type': 'MOUSE_MOVE',
                'event_args': args
            })
        else:
            # Keep other events as is
            processed_rows.append({
                'frame_number': row['frame_number'],
                'event_type': row['event_type'],
                'event_args': row['event_args']
            })
    
    # Create new dataframe
    df = pd.DataFrame(processed_rows)
    
    # Aggregate mouse movements by frame
    mouse_moves = df[df['event_type'] == 'MOUSE_MOVE'].copy()
    other_events = df[df['event_type'] != 'MOUSE_MOVE'].copy()
    
    if len(mouse_moves) > 0:
        # Group by frame and take the mean of the movements
        aggregated_moves = []
        for frame, group in mouse_moves.groupby('frame_number'):
            mean_x = sum(args[0] for args in group['event_args']) / len(group)
            mean_y = sum(args[1] for args in group['event_args']) / len(group)
            
            aggregated_moves.append({
                'frame_number': frame,
                'event_type': 'MOUSE_MOVE',
                'event_args': np.array([mean_x, mean_y], dtype=float)
            })
        
        # Combine aggregated moves with other events
        aggregated_df = pd.DataFrame(aggregated_moves)
        df = pd.concat([other_events, aggregated_df], ignore_index=True)
    else:
        df = other_events
    
    # Sort by frame number
    df = df.sort_values('frame_number').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    path = "overwatch/inputs.csv"
    df = process_controls(path)
    df.to_csv("overwatch/inputs_processed.csv", index=False)