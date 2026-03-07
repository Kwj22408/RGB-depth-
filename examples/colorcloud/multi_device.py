# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

from queue import Queue
from typing import List

import cv2
import numpy as np

from pyorbbecsdk import *
from utils import frame_to_bgr_image

# ====================== 自定义分辨率配置（在这里修改）======================
MAX_DEVICES = 1  # 最多支持的相机数量，可修改
# 自定义RGB分辨率 (宽度, 高度)，设为(0,0)则使用默认分辨率
CUSTOM_COLOR_RES = (1920, 1080)  # 示例：1280x720、640x480、800x600等
# 自定义深度分辨率 (宽度, 高度)，设为(0,0)则使用默认分辨率
CUSTOM_DEPTH_RES = (1920, 1080)
# 帧率配置（设为0则使用默认帧率）
CUSTOM_FPS = 30
# ==========================================================================

curr_device_cnt = 0

MAX_QUEUE_SIZE = 5
ESC_KEY = 27

color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
has_color_sensor: List[bool] = [False for _ in range(MAX_DEVICES)]
stop_rendering = False


def on_new_frame_callback(frames: FrameSet, index: int):
    global color_frames_queue, depth_frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is not None:
        if color_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            color_frames_queue[index].get()
        color_frames_queue[index].put(color_frame)
    if depth_frame is not None:
        if depth_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            depth_frames_queue[index].get()
        depth_frames_queue[index].put(depth_frame)


def rendering_frames():
    global color_frames_queue, depth_frames_queue
    global curr_device_cnt
    global stop_rendering
    while not stop_rendering:
        for i in range(curr_device_cnt):
            color_frame = None
            depth_frame = None
            if not color_frames_queue[i].empty():
                color_frame = color_frames_queue[i].get()
            if not depth_frames_queue[i].empty():
                depth_frame = depth_frames_queue[i].get()
            if color_frame is None and depth_frame is None:
                continue
            color_image = None
            depth_image = None
            color_width, color_height = 0, 0
            if color_frame is not None:
                color_width, color_height = color_frame.get_width(), color_frame.get_height()
                color_image = frame_to_bgr_image(color_frame)
            if depth_frame is not None:
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue

                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))
                except ValueError:
                    print("Failed to reshape depth data")
                    continue

                depth_data = depth_data.astype(np.float32) * scale

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            if color_image is not None and depth_image is not None:
                window_size = (color_width // 2, color_height // 2)
                color_image = cv2.resize(color_image, window_size)
                depth_image = cv2.resize(depth_image, window_size)
                image = np.hstack((color_image, depth_image))
            elif depth_image is not None and not has_color_sensor[i]:
                image = depth_image
            else:
                continue
            cv2.imshow("Device {}".format(i), image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                stop_rendering = True
                break
    cv2.destroyAllWindows()


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print("Starting device {}".format(index))
        pipeline.start(config, lambda frame_set, curr_index=index: on_new_frame_callback(frame_set,
                                                                                         curr_index))
        index += 1


def stop_streams(pipelines: List[Pipeline]):
    for pipeline in pipelines:
        pipeline.stop()


def get_custom_stream_profile(profile_list: StreamProfileList, width: int, height: int, fps: int,
                              sensor_type: OBSensorType):
    """
    获取自定义分辨率的流配置，若不支持则返回默认配置
    :param profile_list: 流配置列表
    :param width: 自定义宽度
    :param height: 自定义高度
    :param fps: 自定义帧率
    :param sensor_type: 传感器类型（COLOR/DEPTH）
    :return: VideoStreamProfile
    """
    profile_type = "color" if sensor_type == OBSensorType.COLOR_SENSOR else "depth"
    try:
        # 优先尝试自定义分辨率+帧率
        if width > 0 and height > 0 and fps > 0:
            print(f"尝试设置{profile_type}分辨率: {width}x{height} @ {fps}fps")
            return profile_list.get_video_stream_profile(width, height,
                                                         OBFormat.RGB if sensor_type == OBSensorType.COLOR_SENSOR else OBFormat.Y16,
                                                         fps)
        # 仅自定义分辨率，使用默认帧率
        elif width > 0 and height > 0:
            print(f"尝试设置{profile_type}分辨率: {width}x{height} (默认帧率)")
            return profile_list.get_video_stream_profile(width, height,
                                                         OBFormat.RGB if sensor_type == OBSensorType.COLOR_SENSOR else OBFormat.Y16,
                                                         0)
        else:
            # 使用默认配置
            print(f"使用{profile_type}默认分辨率")
            return profile_list.get_default_video_stream_profile()
    except OBError as e:
        print(f"不支持{profile_type}分辨率 {width}x{height}，使用默认配置: {e}")
        return profile_list.get_default_video_stream_profile()


def main():
    ctx = Context()
    device_list = ctx.query_devices()
    global curr_device_cnt
    curr_device_cnt = device_list.get_count()
    if curr_device_cnt == 0:
        print("No device connected")
        return
    if curr_device_cnt > MAX_DEVICES:
        print(f"连接的设备数({curr_device_cnt})超过最大支持数({MAX_DEVICES})")
        return
    pipelines: List[Pipeline] = []
    configs: List[Config] = []
    global has_color_sensor

    # 打印自定义配置信息
    print(f"\n自定义配置：")
    print(f"RGB分辨率: {CUSTOM_COLOR_RES[0]}x{CUSTOM_COLOR_RES[1]} (0x0表示默认)")
    print(f"深度分辨率: {CUSTOM_DEPTH_RES[0]}x{CUSTOM_DEPTH_RES[1]} (0x0表示默认)")
    print(f"帧率: {CUSTOM_FPS} (0表示默认)\n")

    for i in range(device_list.get_count()):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        config = Config()
        try:
            # 获取彩色流配置（自定义分辨率）
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile: VideoStreamProfile = get_custom_stream_profile(
                profile_list, CUSTOM_COLOR_RES[0], CUSTOM_COLOR_RES[1], CUSTOM_FPS, OBSensorType.COLOR_SENSOR
            )
            config.enable_stream(color_profile)
            has_color_sensor[i] = True
            print(
                f"设备{i} RGB流配置: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")
        except OBError as e:
            print(f"设备{i} 不支持彩色流: {e}")
            has_color_sensor[i] = False

        # 获取深度流配置（自定义分辨率）
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile: VideoStreamProfile = get_custom_stream_profile(
                profile_list, CUSTOM_DEPTH_RES[0], CUSTOM_DEPTH_RES[1], CUSTOM_FPS, OBSensorType.DEPTH_SENSOR
            )
            config.enable_stream(depth_profile)
            print(
                f"设备{i} 深度流配置: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
        except OBError as e:
            print(f"设备{i} 获取深度流配置失败: {e}")
            continue

        pipelines.append(pipeline)
        configs.append(config)

    global stop_rendering
    start_streams(pipelines, configs)
    try:
        rendering_frames()
    except KeyboardInterrupt:
        stop_rendering = True
    finally:
        stop_streams(pipelines)
        cv2.destroyAllWindows()
        print("\n程序已退出，资源已释放")


if __name__ == "__main__":
    main()