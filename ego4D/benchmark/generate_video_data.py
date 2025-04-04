import os
import json
import subprocess

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"

def process_annotations(subset):
    annotations_file_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', subset)

    print("Loading json file. ")
    with open(annotations_file_path, 'r') as f:
        jsonfile = json.load(f)
    print("File loaded succesfully. ")

    clip_data = []
    video_count = 0
    for video_item in jsonfile['videos']:
        for clip_item in video_item['clips']:
            clip_uid = clip_item['clip_uid']
            print(f"Clip UID: {clip_uid}")
            frame_with_person = {}  # Initialize

            persons = clip_item['persons']
            for person in persons:
                if not person['camera_wearer']:
                    person_id = person['person_id']
                    frame_with_person[person_id] = []

                    tracking_paths = person['tracking_paths']
                    for tracking_path in tracking_paths:
                        frame_start = tracking_path['track'][0]["frame"]
                        frame_end = tracking_path['track'][-1]["frame"]
                        time_start = frame_start / 30
                        time_end = frame_end / 30
                        frame_with_person[person_id].append((time_start, time_end))

            time_person_looking = {} # Initialize
            segments_looking = clip_item['social_segments_looking']
            for item in segments_looking:
                person_id = item['person']
                time_start = item['start_time']
                time_end = item['end_time']
                if time_end - time_start > 3:
                    video_count+=1
                    cut_video(time_start, time_end, clip_uid, person_id)
                    # input()
        if video_count > 10:
            break
            
                # if person_id not in time_person_looking:
                #     time_person_looking[person_id] = [(item['start_time'], item['end_time'])]
                # else:
                #     time_person_looking[person_id].append((item['start_time'], item['end_time']))
            
            # print(f"frames: ")
            # print(json.dumps(frame_with_person, indent=4))
            # print(f"times: ")
            # print(json.dumps(time_person_looking, indent=4))


def cut_video(time_start, time_end, clip_uid, person_id):
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'v2', 'clips')
    new_clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset')

    input_path = os.path.join(clips_directory, clip_uid + ".mp4")
    outputfile = f"{clip_uid}_{person_id}_{int(time_start)}.mp4"
    output_path = os.path.join(new_clips_directory, outputfile)

    command = f"ffmpeg -ss {time_start} -to {time_end} -i {input_path} -c copy {output_path}"
    subprocess.call( command, shell=True)



if __name__ == "__main__":
    annotations = process_annotations(subset = "av_train.json")