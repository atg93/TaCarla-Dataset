import cv2
import numpy as np
import os

splits = ["train", "val", "test"]
main_bdd100k_frame_videos = "/data/ssd/bdd100k/video_frames"
main_bdd100k_video_folder = "/datasets/bdd100k/videos/"
main_bdd100k_image_folder = "/datasets/bdd100k/images/100k/"
# start_index = 69059 is problematic
start_index = 0
for split in splits:
    save_video_images_folder = os.path.join(main_bdd100k_frame_videos, split)
    if not os.path.exists(save_video_images_folder):
        os.makedirs(save_video_images_folder)

    video_folder = os.path.join(main_bdd100k_video_folder, split)
    image_folder = os.path.join(main_bdd100k_image_folder, split)
    video_files = os.listdir(video_folder)
    # video_files = ["8a3c3d7c-fd4fb0c8.mov"]
    for video_index, video_file in enumerate(video_files[start_index:]):
        print(f"{video_file} started, index: {video_index+start_index}")
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        image = cv2.imread(os.path.join(image_folder, f"{video_file[:-3]}jpg"))
        frame_num = 0
        frame_list = []
        min_distance = 100
        min_frame_num = None
        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            frame_list.append(frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == ord('p'):
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('p'):
                        break
            if frame_num > 330:
                break

            # cv2.imshow(winname="bdd_video", mat=frame)
            # cv2.imshow(winname="bdd_image", mat=image)

            distance = np.mean(np.abs(frame.astype(float) - image.astype(float)))
            if distance < min_distance:
                min_distance = distance
                min_frame_num = frame_num

            # print(f"frame num: {frame_num}, difference: {distance}")
            frame_num += 1

        if frame_num < 250:
            print(f"Erronous Video File {video_file}")
            continue
        video_file_folder = os.path.join(save_video_images_folder, video_file[:-4])
        if not os.path.exists(video_file_folder):
            os.makedirs(video_file_folder)

        for i in range(-63, 1, 1):
            cv2.imwrite(filename=f"{video_file_folder}/frame_{-i:03d}.jpg", img=frame_list[min_frame_num+i])

        print(f"{video_file} completed")
