import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('/home/sw/probono/best.pt')

# 이미지 로드
image_path = '/home/sw/probono/yb3.jpg'
image = cv2.imread(image_path)

# 모델 추론
results = model.predict(source=image, save=False, show=False)

# 객체 수와 폴리곤 좌표 정보 출력
num_objects = len(results[0].masks.data)
print(f'Number of detected objects: {num_objects}')  # 추론한 대상의 수

all_polygons = [results[0].masks.xy[i] for i in range(num_objects)]

# 폴리곤 병합 함수 (예시)
def merge_polygons(polygons, min_distance=10):
    merged_polygons = []
    for poly in polygons:
        merged = False
        for i, m_poly in enumerate(merged_polygons):
            if np.linalg.norm(m_poly[-1] - poly[0]) < min_distance:
                merged_polygons[i] = np.vstack([m_poly, poly])
                merged = True
                break
        if not merged:
            merged_polygons.append(poly)
    return merged_polygons

# 폴리곤 병합 적용
merged_polygons = merge_polygons(all_polygons)

for polygon_points in merged_polygons:
    polygon_points = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

# 결과 이미지 저장 및 출력
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
