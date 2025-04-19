import os
import json
import subprocess
import pandas as pd

from pathlib import Path
 
if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"

def is_valid_interaction(text):
    unuseful_actions = ['stands besides',
                        'stands beside',
                        'interacts with',
                        'looks',
                        'looks at',
                        # 'talks with', <- Fine as long as person is within image
                        # 'converses with', <- Fine as long as person is within image
                        # 'talks to', <- Fine as long as person is within image
                        # 'speaks to ' <- Fine as long as person is within image
                        # '#unsure', # <- Could be removed if enough samples
                        ]

    if any(action in text for action in unuseful_actions):
        return False

    if "#C" in text and "#O" in text:
        ego_ind = text.find("#C")
        oth_ind = text.find("#O")
        if oth_ind < ego_ind:
            return True

    return False

def find_interactions(jsonfile):
    num_interactions = 0
    video_list = []
    video_uids = jsonfile.keys()
    for video_uid in video_uids:
        annotations = jsonfile[video_uid]
        for key in annotations.keys():
            if 'narration' in key:
                narrations = annotations[key]['narrations']
                for narration in narrations:
                    text = narration['narration_text']
                    if is_valid_interaction(text):
                        num_interactions+=1
                        if video_uid not in video_list:
                            video_list.append(video_uid)

    dict = {'video_uids': video_list}
    df = pd.DataFrame(dict)
    df.to_csv('temp_video_list.csv', index=False)
    print(f"\nTotal of interactions: {num_interactions}")

def extract_frames_and_clip(time, video_uid, output_uid):
    time_start = time - 2
    time_end =  time + 2
    original_clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'v2', 'full_scale')
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'clips')

    video_input_path = os.path.join(original_clips_directory, video_uid + ".mp4")
    video_output_path = os.path.join(clips_directory, output_uid + ".mp4")

    if not Path(video_output_path).exists():
        command = f"ffmpeg -i {video_input_path} -ss {time_start} -to {time_end} -c copy {video_output_path}"
        subprocess.call(command, shell=True)

    times = [time - 0.5, time, time + 0.5]
    frames_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'frames')
    video_input_path = os.path.join(original_clips_directory, video_uid + ".mp4")

    for n, t in enumerate(times):
        frames_output_path = os.path.join(frames_directory, f"{output_uid}_{n}.png")
        if not Path(frames_output_path).exists():
            command = f"ffmpeg -ss {t} -i {video_input_path} -frames:v 1 {frames_output_path}"
            subprocess.call(command, shell=True)

def extract_interactions_from_videos(jsonfile):
    videos_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'v2', 'full_scale')    
    original_video_uids = [f.split(".mp4")[0] for f in os.listdir(videos_path) if f.endswith(".mp4")]

    video_uids = []
    video_labels = []
    video_times = []
    video_frames = []
    video_natarrion = []
    video_summary = []
    for video_uid in original_video_uids:
        if video_uid not in jsonfile:
            print(f"Video UID {video_uid} not found in the JSON file.")
    
        # Get the annotations for the current video UID
        annotations = jsonfile[video_uid]
        
        for key in annotations.keys():
            if 'narration_pass' in key:
                summaries = annotations[key]['summaries']
                narrations = annotations[key]['narrations']
                for narration in narrations:
                    time = narration['timestamp_sec']
                    frame = narration['timestamp_frame']
                    text = narration['narration_text']
                    if is_valid_interaction(text):
                        summary = ""
                        for summarie in summaries:
                            if time > summarie['start_sec'] and time < summarie['end_sec']:
                                summary = summarie['summary_text']

                        output_uid = f"{video_uid}_{str(frame)}"
                        extract_frames_and_clip(time, video_uid, output_uid)
                        video_uids.append(video_uid)
                        video_labels.append(output_uid)
                        video_times.append(time)
                        video_frames.append(frame)
                        video_natarrion.append(text)
                        video_summary.append(summary)
                        print(f"Processed clip: {video_uid}_{int(time-2)}")
                        # input("Press ENTER to continue:")

    data_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'data.csv')
    dict = {'video_uid': video_uids, 
            'clip_uid': video_labels, 
            'time': video_times, 
            'frame': video_frames, 
            'narration': video_natarrion, 
            'summary': video_summary}
    df_data = pd.DataFrame(dict)
    df_data.to_csv(data_path, index=False)

if __name__ == "__main__":
    print("Loading narration.json file. ")
    annotations_file_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'v2', 'annotations', 'narration.json')
    with open(annotations_file_path, 'r') as f:
        narration_jsonfile = json.load(f)
    print("File loaded succesfully. ")

    find_interactions(narration_jsonfile)
    extract_interactions_from_videos(narration_jsonfile)