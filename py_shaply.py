import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import random

# YOLOv8 모델 로드
model_op = YOLO('/home/sw/probono/op.pt')
model_yb = YOLO('/home/sw/probono/yb.pt')

# 이미지 로드
image_path = '/home/sw/probono/yb3.jpg'
image = cv2.imread(image_path)

# 이미지 크기를 640x640으로 조정
image_resized = cv2.resize(image, (640, 640))

# 모델 추론
results_op = model_op.predict(source=image_resized, save=False, show=False)
results_yb = model_yb.predict(source=image_resized, save=False, show=False)

# 두 모델의 결과 병합
masks_op_data = results_op[0].masks.data.cpu().numpy()
masks_yb_data = results_yb[0].masks.data.cpu().numpy()
boxes_op_cls = results_op[0].boxes.cls.cpu().numpy()
boxes_yb_cls = results_yb[0].boxes.cls.cpu().numpy()
boxes_op_conf = results_op[0].boxes.conf.cpu().numpy()
boxes_yb_conf = results_yb[0].boxes.conf.cpu().numpy()

# 병합된 결과 저장
merged_masks_data = np.concatenate((masks_op_data, masks_yb_data))
merged_boxes_cls = np.concatenate((boxes_op_cls, boxes_yb_cls))
merged_boxes_conf = np.concatenate((boxes_op_conf, boxes_yb_conf))

# 객체 수와 폴리곤 좌표 정보 출력
num_objects = len(merged_masks_data)
print(f'Number of detected objects: {num_objects}')  # 추론한 대상의 수

# 클래스별 폴리곤 저장
polygons_by_class = {}

# 객체마다 색상 정의
object_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_objects)]

for i, (mask, cls, conf) in enumerate(zip(merged_masks_data, merged_boxes_cls, merged_boxes_conf)):
    cls = int(cls)  # 클래스 ID를 정수형으로 변환
    print(f'\nObject {i+1}:')
    print(f'Class: {cls}, Confidence: {conf:.2f}')  # 클래스 및 신뢰도
    polygon_points = results_op[0].masks.xy[i] if i < len(results_op[0].masks.xy) else results_yb[0].masks.xy[i - len(results_op[0].masks.xy)]  # 폴리곤 좌표 추출
    print('Polygon points:', polygon_points)

    # OpenCV로 폴리곤 그리기
    polygon_points = np.array(polygon_points, dtype=np.int32)
    color = object_colors[i]  # 객체마다 고유한 색상 선택
    cv2.polylines(image_resized, [polygon_points], isClosed=True, color=color, thickness=2)

    # Shapely 폴리곤 생성
    poly = Polygon(polygon_points)
    
    # 클래스별 폴리곤 저장
    if cls not in polygons_by_class:
        polygons_by_class[cls] = []
    polygons_by_class[cls].append(poly)

# 클래스별로 폴리곤 겹침 여부 판단
for cls, polygons in polygons_by_class.items():
    print(f'\nClass {cls} polygons overlap check:')
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                print(f'Polygon {i+1} and Polygon {j+1} are overlapping.')

# 결과 이미지 저장
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image_resized)

# 결과 이미지 보기
cv2.imshow('Result', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
