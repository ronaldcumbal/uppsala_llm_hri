import os
import shutil
import random
from pathlib import Path

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"


def organize_videos(source_dir, videos_per_folder=10, output_base="batch", list_filename="video_list.txt"):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    
    # Convert to Path object
    source = Path(source_dir)

    # Get all video files and randomize
    all_videos = [f for f in source.iterdir() if f.suffix.lower() in video_extensions and f.is_file()]
    random.shuffle(all_videos)
    
    # Calculate number of folders
    total_files = len(all_videos)
    num_folders = (total_files + videos_per_folder - 1) // videos_per_folder  # ceiling division
    
    # Create output text file
    list_file_path = source / list_filename
    with list_file_path.open("w", encoding="utf-8") as list_file:
        for i in range(num_folders):
            folder_name = f"{output_base}_{i+1}"
            folder_path = source / folder_name
            folder_path.mkdir(exist_ok=True)
            
            chunk = all_videos[i*videos_per_folder : (i+1)*videos_per_folder]
            
            list_file.write(f"{folder_name}:\n")
            
            for video in chunk:
                target_path = folder_path / video.name
                shutil.move(str(video), str(target_path))
                list_file.write(f"  {video.name}\n")
            
            list_file.write("\n")
    
    print(f"Done! {num_folders} folders created. File list saved to '{list_file_path}'.")


def restore_videos(source_dir, output_base="batch", list_filename="video_list.txt"):
    source = Path(source_dir)
    moved_folders = [f for f in source.iterdir() if f.is_dir() and f.name.startswith(output_base)]

    for folder in moved_folders:
        for file in folder.iterdir():
            if file.is_file():
                shutil.move(str(file), str(source / file.name))
        folder.rmdir()  # Delete empty folder

    # Remove the list file if it exists
    list_file_path = source / list_filename
    if list_file_path.exists():
        list_file_path.unlink()

    print(f"Restoration complete. All videos moved back to '{source_dir}' and folders removed.")


if __name__ == "__main__":
    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset', 'clips')
    # organize_videos(clips_directory, videos_per_folder=17)
    # restore_videos(clips_directory)

    clips_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'clips')
    # organize_videos(clips_directory, videos_per_folder=3)
    # restore_videos(clips_directory)