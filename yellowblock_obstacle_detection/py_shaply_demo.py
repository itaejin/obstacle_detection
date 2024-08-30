import cv2
import numpy as np
from ultralytics import YOLO
import random

# YOLOv8 모델 로드
model_op = YOLO('/home/sw/probono/ob.pt')
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

# 경계 상자를 계산하는 함수
def bounding_box(polygon):
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    return min_x, max_x, min_y, max_y

# 두 경계 상자가 겹치는지 확인하는 함수
def bounding_boxes_intersect(bbox1, bbox2):
    return not (bbox1[1] < bbox2[0] or bbox1[0] > bbox2[1] or bbox1[3] < bbox2[2] or bbox1[2] > bbox2[3])

# 두 선분이 교차하는지 확인하는 함수
def edge_intersection(p1, p2, q1, q2):
    # 세 점의 방향을 계산하는 내부 함수
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # 일직선
        return 1 if val > 0 else 2  # 시계 방향 또는 반시계 방향

    # 점 q가 선분 pr 상에 있는지 확인하는 함수
    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # 일반적인 경우
    if o1 != o2 and o3 != o4:
        return True

    # 특수한 경우 (선분이 일직선 상에 있는 경우)
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

# 두 폴리곤이 교차하는지 확인하는 함수
def polygons_intersect(polygon1, polygon2):
    # 폴리곤의 경계 상자 계산
    bbox1 = bounding_box(polygon1)
    bbox2 = bounding_box(polygon2)

    # 경계 상자가 겹치지 않으면 폴리곤도 겹치지 않음
    if not bounding_boxes_intersect(bbox1, bbox2):
        return False

    # 모든 선분 쌍의 교차 여부 검사
    for i in range(len(polygon1)):
        for j in range(len(polygon2)):
            p1, p2 = polygon1[i], polygon1[(i + 1) % len(polygon1)]
            q1, q2 = polygon2[j], polygon2[(j + 1) % len(polygon2)]
            if edge_intersection(p1, p2, q1, q2):
                return True

    # 한 폴리곤의 점이 다른 폴리곤 안에 있는지 확인하는 내부 함수
    def point_in_polygon(point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # 한 폴리곤의 점이 다른 폴리곤 안에 있는지 확인
    for point in polygon1:
        if point_in_polygon(point, polygon2):
            return True

    for point in polygon2:
        if point_in_polygon(point, polygon1):
            return True

    return False

# 객체마다 색상 정의 및 폴리곤 그리기
for i, (mask, cls, conf) in enumerate(zip(merged_masks_data, merged_boxes_cls, merged_boxes_conf)):
    cls = int(cls)  # 클래스 ID를 정수형으로 변환
    print(f'\nObject {i+1}:')
    print(f'Class: {cls}, Confidence: {conf:.2f}')  # 클래스 및 신뢰도
    # 폴리곤 좌표 추출
    polygon_points = results_op[0].masks.xy[i] if i < len(results_op[0].masks.xy) else results_yb[0].masks.xy[i - len(results_op[0].masks.xy)]
    print('Polygon points:', polygon_points)

    # OpenCV로 폴리곤 그리기
    polygon_points = np.array(polygon_points, dtype=np.int32)
    color = object_colors[i]  # 객체마다 고유한 색상 선택
    cv2.polylines(image_resized, [polygon_points], isClosed=True, color=color, thickness=2)

    # 클래스별 폴리곤 저장
    if cls not in polygons_by_class:
        polygons_by_class[cls] = []
    polygons_by_class[cls].append(polygon_points)

# 클래스별로 폴리곤 겹침 여부 판단
for cls, polygons in polygons_by_class.items():
    print(f'\nClass {cls} polygons overlap check:')
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons_intersect(polygons[i], polygons[j]):
                print(f'Polygon {i+1} and Polygon {j+1} are overlapping.')

# 결과 이미지 저장
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image_resized)

# 결과 이미지 보기
cv2.imshow('Result', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
