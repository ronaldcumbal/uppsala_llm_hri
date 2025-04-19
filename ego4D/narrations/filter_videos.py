import os
import cv2
import torch
import pandas as pd

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"


def detect_person_in_video(clips_directory, log_file, confidence_threshold=0.3):

    # Load YOLOv5 model from PyTorch Hub (you can replace with local weights if needed)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    for video in os.listdir(clips_directory):
        video_path = os.path.join(clips_directory, video)
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_index = 0

        person_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # No more frames

            frame_index += 1

            # Convert BGR (OpenCV) to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv5 inference
            results = model(img)

            # Parse results
            detections = results.xyxy[0]  # Bounding boxes (x1, y1, x2, y2, conf, class)

            # Check for person (class 0 in COCO dataset)
            person_detected = any(det[5] == 0 and det[4] >= confidence_threshold for det in detections)

            if person_detected:
                break

        cap.release()
        cv2.destroyAllWindows()

        if not person_detected:
            with open(log_file, 'a') as f:
                f.write(f"{video}\n")


def verify_video_length(clips_directory, log_file, min_duration=3.0):

    for video in os.listdir(clips_directory):
        video_path = os.path.join(clips_directory, video)
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        # if not cap.isOpened():
        #     print(f"Could not open video: {video_path}")
        #     return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration < min_duration:
            with open(log_file, 'a') as f:
                f.write(f"{video}\n")
            print(f"Video too short ({duration:.2f} sec): Logged {video_path}")


def remove_videos(clips_directory, log_file):
    
    with open(log_file, 'r') as f:
        video_names = [line.strip() for line in f if line.strip()]

    deleted_clip_uids = []
    for video in video_names:
        
        clip_uids = video.split('.mp4')[0]
        deleted_clip_uids.append(clip_uids)

        video_path = os.path.join(clips_directory, video)
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted: {video}")
            except Exception as e:
                print(f"Error deleting {video}: {e}")
        else:
            print(f"File not found: {video}")

    data_path = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'data.csv')
    df_data = pd.read_csv(data_path)  

    original_len = len(df_data)
    df_data = df_data[~df_data["clip_uid"].isin(deleted_clip_uids)]
    removed = original_len - len(df_data)

    df_data.to_csv(data_path, index=False)

    print(f"Removed {removed} rows from DataFrame.")


if __name__ == "__main__":
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'clips')
    log_file = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'remove_videos.txt')

    # detect_person_in_video(clips_directory, log_file)
    # verify_video_length(clips_directory, log_file, min_duration=3.0)

    remove_videos(clips_directory, log_file)
