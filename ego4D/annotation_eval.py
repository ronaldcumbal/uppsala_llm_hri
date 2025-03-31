import os
import json

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"

def complete_annotations():
    annotations_file_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'data', 'ego4d.json')
    with open(annotations_file_path, 'r') as f:
        jsonfile = json.load(f)
    clip_annotations = jsonfile['clips']
    # clip_uids = [item['clip_uid'] for item in clip_annotations]
    return clip_annotations

def benchmark_annotations(subset):
    print("Loading json file. ")
    annotations_file_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', subset)
    with open(annotations_file_path, 'r') as f:
        jsonfile = json.load(f)
    print("File loaded succesfully. ")

    clip_data = []
    for video_item in jsonfile['videos']:
        # print(f"Video UID: {video_item['video_uid']}")
        for clip_item in video_item['clips']:
            # persons = clip_item['persons']
            transcriptions = clip_item['transcriptions']
            talking = clip_item['social_segments_talking']
            looking = clip_item['social_segments_looking']
            print(f"    Clip UID: {clip_item['clip_uid']}")
            print(json.dumps(transcriptions, indent=4))
            input()

            clip_data.append(clip_item)

    return clip_data

def review_annotations(clip_videnames, annotations):
    for item in annotations:
        clip_uid = item["clip_uid"]
        if clip_uid not in clip_videnames:
            print(f"Video clip {clip_uid} not found.")
            continue
        else:
            print(json.dumps(item, indent=4))
            input()

if __name__ == "__main__":
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'data', 'v2', 'clips')
    clip_videnames = [f.split(".")[0] for f in os.listdir(clips_directory) if f.endswith(".mp4")]
    clip_videnames.sort()

    # annotations = complete_annotations()
    annotations = benchmark_annotations(subset = "av_val.json")

    review_annotations(clip_videnames, annotations)