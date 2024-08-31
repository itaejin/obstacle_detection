import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드 (점자블록만 감지)
model_yb = YOLO(r'/home/sw/yellowblock_obstacle_detection/yb.pt')

# 동영상 로드
video_path = r'/home/sw/yellowblock_obstacle_detection/include/probono.mp4'  # 동영상 경로
cap = cv2.VideoCapture(video_path)

# 동영상이 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 첫 번째 프레임 읽기
ret, frame = cap.read()

# 동영상이 비어 있는지 확인
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# 첫 번째 프레임의 크기를 640x640으로 조정
frame_resized = cv2.resize(frame, (640, 640))

# 모델 추론
results_yb = model_yb.predict(source=frame_resized, save=False, show=False)

# 점자블록의 폴리곤 좌표 추출
polygons = []
for i in range(len(results_yb[0].masks.xy)):
    polygon_points = results_yb[0].masks.xy[i]
    polygons.append(polygon_points)

    # OpenCV로 폴리곤 그리기
    polygon_points = np.array(polygon_points, dtype=np.int32)  # 정수형으로 변환
    cv2.polylines(frame_resized, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

# 점자블록의 폴리곤 좌표를 텍스트 파일에 저장
output_txt_path = 'braille_blocks_coordinates.txt'
with open(output_txt_path, 'w') as f:
    for polygon in polygons:
        for point in polygon:
            f.write(f'{point[0]},{point[1]}\n')
        f.write('\n')  # 폴리곤 구분을 위해 빈 줄 추가

print(f'Braille block coordinates saved to {output_txt_path}')

# 결과 이미지 보기
cv2.imshow('Detected Braille Blocks', frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 동영상 객체 해제
cap.release()
