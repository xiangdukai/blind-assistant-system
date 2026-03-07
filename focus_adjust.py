"""
调焦辅助工具：同时显示左右两路原始画面，帮助手动调节双目摄像头焦距
操作：
  q   退出
  s   截图保存（左右分别保存）
"""
import sys, os, cv2
import numpy as np

CAMERA_INDEX = 0
TARGET_W, TARGET_H = 1920, 1080   # 单眼期望分辨率

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
eye_w = actual_w // 2
print(f"实际总分辨率: {actual_w}x{actual_h}，每眼: {eye_w}x{actual_h}")
print("按 q 退出，按 s 截图")

screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("读帧失败")
        continue

    left  = frame[:, :eye_w, :]
    right = frame[:, eye_w:eye_w*2, :]

    # 缩放到显示尺寸（屏幕可能放不下 1920 宽）
    disp_w = min(eye_w, 960)
    disp_h = int(actual_h * disp_w / eye_w)
    left_d  = cv2.resize(left,  (disp_w, disp_h))
    right_d = cv2.resize(right, (disp_w, disp_h))

    # 添加标签
    cv2.putText(left_d,  "LEFT",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(right_d, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)

    display = np.hstack([left_d, right_d])
    cv2.imshow('Focus Adjust - LEFT | RIGHT  (q=quit, s=screenshot)', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        frame_count += 1
        os.makedirs(screenshot_dir, exist_ok=True)
        cv2.imwrite(os.path.join(screenshot_dir, f'left_{frame_count:04d}.jpg'), left)
        cv2.imwrite(os.path.join(screenshot_dir, f'right_{frame_count:04d}.jpg'), right)
        print(f"截图已保存: left_{frame_count:04d}.jpg / right_{frame_count:04d}.jpg")

cap.release()
cv2.destroyAllWindows()
print("退出")
