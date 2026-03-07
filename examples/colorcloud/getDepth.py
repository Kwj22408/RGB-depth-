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
import time
import os
import cv2
import numpy as np

from pyorbbecsdk import *

ESC_KEY = 27
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def main():
    config = Config()
    pipeline = Pipeline()
    frame_count = 0
    save_dir = "depth_data_uint16"
    os.makedirs(save_dir, exist_ok=True)

    print("正在配置深度流...")

    try:
        # 方式1：使用配置文件方式（根据你的SDK版本）
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if profile_list is not None:
            print("找到深度流配置列表")

            # 查找512x512的配置
            depth_profile = None

            # 方法1：使用迭代器方式（如果你的SDK支持）
            try:
                # 尝试获取配置数量
                count = profile_list.get_count()
                print(f"共有 {count} 个配置")

                for i in range(count):
                    try:
                        profile = profile_list.get_profile(i)
                        if hasattr(profile, 'as_video_stream_profile'):
                            video_profile = profile.as_video_stream_profile()
                            if (video_profile.get_width() == TARGET_WIDTH and
                                    video_profile.get_height() == TARGET_HEIGHT):
                                depth_profile = video_profile
                                print(f"找到目标配置 {TARGET_WIDTH}x{TARGET_HEIGHT}")
                                break
                    except:
                        continue
            except:
                # 方法2：尝试使用默认配置
                print("使用默认视频流配置")
                depth_profile = profile_list.get_default_video_stream_profile()

            if depth_profile:
                config.enable_stream(depth_profile)
                print(f"使用配置: {depth_profile}")
            else:
                print("未找到512x512配置，启用默认深度传感器")
                config.enable_stream(OBSensorType.DEPTH_SENSOR)
        else:
            print("无法获取配置列表，启用默认深度传感器")
            config.enable_stream(OBSensorType.DEPTH_SENSOR)

    except Exception as e:
        print(f"配置时出错: {e}")
        print("尝试启用默认深度传感器...")
        config.enable_stream(OBSensorType.DEPTH_SENSOR)

    print("启动管道...")
    pipeline.start(config)

    print("\n" + "=" * 50)
    print("深度数据采集程序")
    print("=" * 50)
    print("操作说明:")
    print("  1. 按 's' 键: 保存当前深度图 (uint16格式)")
    print("  2. 按 'q' 键: 退出程序")
    print(f"  保存目录: {save_dir}")
    print("=" * 50 + "\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                print("等待帧...")
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print("无深度帧")
                continue

            # 获取深度数据
            width = depth_frame.get_width()
            height = depth_frame.get_height()

            # 打印当前分辨率
            if frame_count == 0:
                print(f"当前深度图分辨率: {width}x{height}")
                print(f"深度格式: {depth_frame.get_format()}")
                print(f"深度单位: {depth_frame.get_depth_scale() * 1000} mm/单位")

            # 获取uint16深度数据
            try:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
            except Exception as e:
                print(f"数据处理错误: {e}")
                continue

            # 如果分辨率不是512x512，进行调整
            if width != TARGET_WIDTH or height != TARGET_HEIGHT:
                print(f"调整分辨率: {width}x{height} -> {TARGET_WIDTH}x{TARGET_HEIGHT}")
                depth_data = cv2.resize(depth_data, (TARGET_WIDTH, TARGET_HEIGHT),
                                        interpolation=cv2.INTER_NEAREST)

            # 显示深度图（灰度）
            # 注意：直接显示uint16数据可能太暗，需要归一化
            depth_vis = depth_data.astype(np.float32)

            # 过滤无效值（通常0是无效值）
            valid_mask = depth_vis > 0
            if np.any(valid_mask):
                valid_depths = depth_vis[valid_mask]
                min_depth = np.percentile(valid_depths, 5)
                max_depth = np.percentile(valid_depths, 95)
            else:
                min_depth, max_depth = 0, 1000

            # 归一化到0-255用于显示
            depth_vis_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
            if max_depth > min_depth:
                depth_vis_normalized = np.clip((depth_vis - min_depth) / (max_depth - min_depth) * 255, 0, 255).astype(
                    np.uint8)

            cv2.imshow("Depth Viewer (512x512)", depth_vis_normalized)

            # 显示信息
            info_img = np.zeros((120, 512, 3), dtype=np.uint8)
            cv2.putText(info_img, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_img, f"Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_img, f"Depth range: {depth_data.min()}-{depth_data.max()} mm",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(info_img, "Press 's' to save, 'q' to quit",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Info", info_img)

            # 等待按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # 保存深度图
                filename = os.path.join(save_dir, f"depth_{frame_count:04d}.png")
                success = cv2.imwrite(filename, depth_data)
                if success:
                    print(f"[{frame_count}] 保存成功: {filename}")
                    print(f"    数据类型: {depth_data.dtype}")
                    print(f"    数据形状: {depth_data.shape}")
                    print(f"    深度范围: {depth_data.min()} - {depth_data.max()} mm")

                    # 验证保存的文件
                    saved_img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
                    if saved_img is not None:
                        print(f"    验证: 已保存 {saved_img.shape} {saved_img.dtype}")
                    else:
                        print(f"    警告: 无法读取保存的文件")

                    frame_count += 1
                else:
                    print(f"保存失败: {filename}")

            elif key == ord('q') or key == ESC_KEY:
                print("退出程序...")
                break

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print(f"\n程序结束，共保存 {frame_count} 帧深度图")
        print(f"数据保存在: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()