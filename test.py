import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('/home/sw/probono/optacle.pt')

# 이미지 로드
image_path = '/home/sw/probono/yb3.jpg'
image = cv2.imread(image_path)

# 이미지 크기를 640x640으로 조정
image_resized = cv2.resize(image, (640, 640))

# 모델 추론
results = model.predict(source=image_resized, save=False, show=False)

# 객체 수와 폴리곤 좌표 정보 출력
print(f'Number of detected objects: {len(results[0].masks.data)}')  # 추론한 대상의 수

for i, (mask, cls, conf) in enumerate(zip(results[0].masks.data, results[0].boxes.cls, results[0].boxes.conf)):
    print(f'\nObject {i+1}:')
    print(f'Class: {int(cls)}, Confidence: {conf:.2f}')  # 클래스 및 신뢰도
    polygon_points = results[0].masks.xy[i]  # 폴리곤 좌표 추출
    print('Polygon points:', polygon_points)

    # OpenCV로 폴리곤 그리기
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.polylines(image_resized, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

# 결과 이미지 저장
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image_resized)

# 결과 이미지 보기
cv2.imshow('Result', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
