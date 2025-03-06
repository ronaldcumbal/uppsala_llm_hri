import os
import cv2

seq = "003"
base_dir = "/home/ronald/Github/uppsala_llm_hri/eu_hri_dataset"
seq_dir = os.path.join(base_dir, "sequences",seq)

video_path = os.path.join(base_dir, "seq"+seq+".avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

file_list = [file for file in os.listdir(seq_dir)]
for img_file in sorted(file_list):
    img_path = os.path.join(seq_dir,img_file)
    img = cv2.imread(img_path)
    video_out.write(img)