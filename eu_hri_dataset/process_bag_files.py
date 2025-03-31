import os
import cv2
from matplotlib import pyplot as plt
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage

def save_frame(img, msg, bag_id, stamp):
    frame_dir = os.path.join("frames", bag_id)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    frame_path = os.path.join(frame_dir, str(msg.header.stamp.sec) + "_" + str(msg.header.stamp.nanosec) + ".jpg") 
    cv2.imwrite(frame_path, img)    # save frame as JPEG file
    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    # input("Press any key ...")


def read_bag_file(bagpath):
    bag_id = os.path.basename(bagpath).split(".")[0]
    # Set up the video writer
    frame_rate = 10
    video_top_path = os.path.join("videos", bag_id + "_top.avi")
    video_bottom_path = os.path.join("videos", bag_id + "_bottom.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_top = cv2.VideoWriter(video_top_path, fourcc, frame_rate, (640, 480))
    video_bottom = cv2.VideoWriter(video_bottom_path, fourcc, frame_rate, (640, 480))
    print(f"Writing video to: \n{video_top_path} \n{video_bottom_path}")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_FOXY)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if msg.__msgtype__ == "sensor_msgs/msg/Image" and msg.header.frame_id == "CameraTop_optical_frame":
                img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                video_top.write(img)
            if msg.__msgtype__ == "sensor_msgs/msg/Image" and msg.header.frame_id == "CameraBottom_optical_frame":
                img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                video_bottom.write(img)
            # save_frame(img, frame_dir, msg.header.stamp)

if __name__ == '__main__':

    avoid_files = ["user1_2017-06-14.bag"]

    bag_dir = Path('dataset')
    file_list = [file for file in bag_dir.iterdir() if file.is_file() and file.suffix == '.bag']
    for file in file_list:
        print("Processing: ", os.path.basename(file))
        if os.path.basename(file) not in avoid_files:
            read_bag_file(file)