import os
import json
import subprocess
import pandas as pd

from pathlib import Path

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"

def process_annotations(annotation_set):
    annotations_file_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', annotation_set)

    print("Loading json file. ")
    with open(annotations_file_path, 'r') as f:
        jsonfile = json.load(f)
    print("File loaded succesfully. ")

    video_uids = []
    video_labels = []
    video_times = []
    video_frames = []
    video_natarrion = []
    video_summary = []

    video_count = 0
    for video_item in jsonfile['videos']:
        for clip_item in video_item['clips']:
            video_uid = clip_item['video_uid']
            clip_uid = clip_item['clip_uid']
            # print(f"Clip UID: {clip_uid}")

            # frame_with_person = {}  # Initialize
            # persons = clip_item['persons']
            # for person in persons:
            #     if not person['camera_wearer']:
            #         person_id = person['person_id']
            #         frame_with_person[person_id] = []

            #         tracking_paths = person['tracking_paths']
            #         for tracking_path in tracking_paths:
            #             frame_start = tracking_path['track'][0]["frame"]
            #             frame_end = tracking_path['track'][-1]["frame"]
            #             time_start = frame_start / 30
            #             time_end = frame_end / 30
            #             frame_with_person[person_id].append((time_start, time_end))

            #         print(f"frames: ")
            #         print(json.dumps(frame_with_person, indent=4))

            time_person_looking = {} # Initialize
            segments_looking = clip_item['social_segments_looking']
            if segments_looking is not None:
                for item in segments_looking:
                    person_id = item['person']
                    time_start = item['start_time']
                    time_end = item['end_time']
                    frame_start = int(time_start * 30)

                    if time_end - time_start > 2:
                        video_count+=1
                        output_uid = f"{clip_uid}_{str(int(time_start))}"
                        extract_frames_and_clip(time_start, clip_uid, output_uid)

                        video_uids.append(clip_uid)
                        video_labels.append(output_uid)
                        video_times.append(time_start)
                        video_frames.append(frame_start)
                        video_natarrion.append("")
                        video_summary.append("")

    dict = {'video_uid': video_uids, 
            'clip_uid': video_labels, 
            'time': video_times, 
            'frame': video_frames, 
            'narration': video_natarrion, 
            'summary': video_summary}
    df_data = pd.DataFrame(dict)
    return df_data

def extract_frames_and_clip(time, video_uid, output_uid):
    time_start = time - 2
    time_end =  time + 2
    original_clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'v2', 'clips')
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset', 'clips')

    video_input_path = os.path.join(original_clips_directory, video_uid + ".mp4")
    video_output_path = os.path.join(clips_directory, output_uid + ".mp4")

    if not Path(video_output_path).exists():
        command = f"ffmpeg -i {video_input_path} -ss {time_start} -to {time_end} -c copy {video_output_path}"
        subprocess.call(command, shell=True)

    times = [time - 0.5, time, time + 0.5]
    frames_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset', 'frames')
    video_input_path = os.path.join(original_clips_directory, video_uid + ".mp4")

    for n, t in enumerate(times):
        frames_output_path = os.path.join(frames_directory, f"{output_uid}_{n}.png")
        if not Path(frames_output_path).exists():
            command = f"ffmpeg -ss {t} -i {video_input_path} -frames:v 1 {frames_output_path}"
            subprocess.call(command, shell=True)

if __name__ == "__main__":
    df_train = process_annotations(annotation_set = "av_train.json")
    df_val = process_annotations(annotation_set = "av_val.json")

    data_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset', 'data.csv')
    df_data = pd.concat([df_train, df_val], ignore_index=True)

    ## Remove duplicates
    df_data = df_data.drop_duplicates(subset=['clip_uid'], keep='last')
    df_data = df_data.reset_index(drop=True)

    df_data.to_csv(data_path, index=False)
