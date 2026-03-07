# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2
import os
from datetime import datetime

from pyorbbecsdk import *
from utils import frame_to_bgr_image

ESC_KEY = 27
# 保存图片的文件夹（自动创建）
SAVE_DIR = "saved_photos"


def main():
    # 创建保存图片的文件夹
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建保存文件夹: {os.path.abspath(SAVE_DIR)}")

    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)
    print("开始采集画面，按 's' 保存图片，按 'q' 或 ESC 退出")

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # 转换为RGB格式图片
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue

            # 显示画面
            cv2.imshow("Color Viewer", color_image)

            # 按键检测（1ms延迟）
            key = cv2.waitKey(1)
            # 按 q 或 ESC 退出
            if key == ord('q') or key == ESC_KEY:
                print("退出程序...")
                break
            # 按 s 保存图片
            elif key == ord('s'):
                # 生成带时间戳的文件名（避免重名）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"color_photo_{timestamp}.png"
                save_path = os.path.join(SAVE_DIR, filename)

                # 保存图片
                cv2.imwrite(save_path, color_image)
                print(f"图片已保存: {os.path.abspath(save_path)}")

        except KeyboardInterrupt:
            print("检测到键盘中断，退出程序...")
            break
        except Exception as e:
            print(f"运行出错: {e}")
            break

    # 释放资源
    cv2.destroyAllWindows()
    pipeline.stop()
    print("程序已退出")


if __name__ == "__main__":
    main()