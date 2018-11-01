#!/bin/bash
cd ./test_video
TARGET_DIR=test_full_frames
mkdir ../${TARGET_DIR}
for video_file_name in *.avi; do
    echo ${video_file_name}
    mkdir ../${TARGET_DIR}/${video_file_name}
    ffmpeg -i ${video_file_name} -f image2 ../${TARGET_DIR}/${video_file_name}/f%06d.jpg
done
