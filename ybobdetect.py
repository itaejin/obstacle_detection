import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드 (장애물 감지)
model_op = YOLO(r'D:\Project\probono\ob.pt')

# 이미지 로드
image_path = r'D:\Project\probono\yb3.jpg'
image = cv2.imread(image_path)

# 이미지 크기를 640x640으로 조정
image_resized = cv2.resize(image, (640, 640))

# 모델 추론 (장애물)
results_op = model_op.predict(source=image_resized, save=False, show=False)

# 저장된 점자블록 좌표 로드
input_txt_path = r'D:\Project\probono\braille_blocks_coordinates.txt'
braille_blocks_polygons = []
with open(input_txt_path, 'r') as f:
    current_polygon = []
    for line in f:
        if line.strip():  # 빈 줄이 아닌 경우
            x, y = map(float, line.strip().split(','))
            current_polygon.append([x, y])
        else:
            if current_polygon:
                braille_blocks_polygons.append(np.array(current_polygon, dtype=np.int32))
                current_polygon = []

# 장애물의 폴리곤 좌표 추출
obstacles_polygons = []
for i in range(len(results_op[0].masks.xy)):
    polygon_points = results_op[0].masks.xy[i]
    obstacles_polygons.append(np.array(polygon_points, dtype=np.int32))

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

# 장애물이 점자블록 위에 있는지 확인
obstacle_on_braille_block = False
for braille_polygon in braille_blocks_polygons:
    for obstacle_polygon in obstacles_polygons:
        if polygons_intersect(braille_polygon, obstacle_polygon):
            obstacle_on_braille_block = True
            break
    if obstacle_on_braille_block:
        break

if obstacle_on_braille_block:
    print("Obstacle is present on the braille block.")
else:
    print("No obstacle on the braille block.")

# OpenCV로 결과 이미지에 점자블록과 장애물 그리기
# 점자블록은 녹색, 장애물은 빨간색으로 표시
for polygon in braille_blocks_polygons:
    cv2.polylines(image_resized, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

for polygon in obstacles_polygons:
    cv2.polylines(image_resized, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

# 결과 이미지 보기
cv2.imshow('Detected Braille Blocks and Obstacles', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()