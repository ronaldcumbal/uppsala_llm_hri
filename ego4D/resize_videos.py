import os
import subprocess

if '/home/ronald' in os.getcwd(): DIRECTORY_PATH = "/home/ronald/Github/uppsala_llm_hri"
elif '/home/roncu858' in os.getcwd(): DIRECTORY_PATH = "/home/roncu858/Github/uppsala_llm_hri"


def resize_videos(video_directory, size_limit_mb):

    # Loop through all files in the directory
    for filename in os.listdir(video_directory):
        if filename.lower().endswith(".mp4"):
            full_path = os.path.join(video_directory, filename)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)

            if size_mb > size_limit_mb:
                print(f"Compressing {filename} (size: {size_mb:.2f} MB)...")

                # Output path (overwrite or create new name)
                compressed_path = os.path.join(video_directory, f"compressed_{filename}")

                # FFmpeg command to compress the video
                command = [
                    "ffmpeg",
                    "-i", full_path,
                    "-vcodec", "libx264",
                    "-crf", "28",  # Adjust for quality vs. compression (lower = better quality)
                    compressed_path
                ]

                # try:
                #     subprocess.run(command, check=True)
                #     print(f"Compressed video saved as {compressed_path}")
                # except subprocess.CalledProcessError as e:
                #     print(f"Error compressing {filename}: {e}")

if __name__ == "__main__":


    video_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'benchmark', 'subset', 'clips')
    resize_videos(video_directory, size_limit_mb = 2)

    video_directory = os.path.join(DIRECTORY_PATH, 'ego4D', 'narrations', 'subset', 'clips')
    resize_videos(video_directory, size_limit_mb = 2)
