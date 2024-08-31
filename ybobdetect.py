import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
import time

# 필요한 함수들 정의
def bounding_box(polygon):
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    return min_x, max_x, min_y, max_y

def bounding_boxes_intersect(bbox1, bbox2):
    return not (bbox1[1] < bbox2[0] or bbox1[0] > bbox2[1] or bbox1[3] < bbox2[2] or bbox1[2] > bbox2[3])

def edge_intersection(p1, p2, q1, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # 일직선
        return 1 if val > 0 else 2  # 시계 방향 또는 반시계 방향

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

def polygons_intersect(polygon1, polygon2):
    bbox1 = bounding_box(polygon1)
    bbox2 = bounding_box(polygon2)

    if not bounding_boxes_intersect(bbox1, bbox2):
        return False

    for i in range(len(polygon1)):
        for j in range(len(polygon2)):
            p1, p2 = polygon1[i], polygon1[(i + 1) % len(polygon1)]
            q1, q2 = polygon2[j], polygon2[(j + 1) % len(polygon2)]
            if edge_intersection(p1, p2, q1, q2):
                return True

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

    for point in polygon1:
        if point_in_polygon(point, polygon2):
            return True

    for point in polygon2:
        if point_in_polygon(point, polygon1):
            return True

    return False

# YOLOv8 모델 로드 (장애물 감지) 및 GPU 사용 설정
model_op = YOLO(r'/home/sw/yellowblock_obstacle_detection/ob.pt')
model_op.to('cuda')  # 'cuda'를 사용하여 GPU를 활성화

# 동영상 로드
video_path = r'/home/sw/yellowblock_obstacle_detection/include/probono.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 동영상의 FPS 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # 프레임 간 지연 시간 계산 (밀리초 단위)

# 저장된 점자블록 좌표 로드
input_txt_path = r'/home/sw/yellowblock_obstacle_detection/braille_blocks_coordinates.txt'
braille_blocks_polygons = []
with open(input_txt_path, 'r') as f:
    current_polygon = []
    for line in f:
        if line.strip():
            x, y = map(float, line.strip().split(','))
            current_polygon.append([x, y])
        else:
            if current_polygon:
                braille_blocks_polygons.append(np.array(current_polygon, dtype=np.int32))
                current_polygon = []

# 프레임 큐와 결과 큐 설정
frame_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=5)

def capture_frames():
    """비디오에서 프레임을 캡처하여 큐에 추가하는 함수"""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            frame_queue.put(None)  # 종료 신호
            break
        frame_resized = cv2.resize(frame, (640, 640))
        frame_queue.put(frame_resized)

def process_frames():
    """큐에서 프레임을 가져와 모델 추론을 수행하는 함수"""
    last_detection_time = time.time()  # 마지막 감지 시간을 기록

    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)  # 종료 신호
            break

        # 현재 시간과 마지막 감지 시간 비교
        current_time = time.time()
        if current_time - last_detection_time >= 5:  # 5초마다 추론 수행
            # 모델 추론 (장애물)
            results_op = model_op.predict(source=frame, device='cuda', save=False, show=False)

            # 장애물의 폴리곤 좌표 추출
            obstacles_polygons = []
            if results_op[0].masks is not None and results_op[0].masks.xy is not None:
                for i in range(len(results_op[0].masks.xy)):
                    polygon_points = results_op[0].masks.xy[i]
                    obstacles_polygons.append(np.array(polygon_points, dtype=np.int32))

            # 장애물이 점자블록 위에 있는지 확인
            obstacle_on_braille_block = False
            for braille_polygon in braille_blocks_polygons:
                for obstacle_polygon in obstacles_polygons:
                    if polygons_intersect(braille_polygon, obstacle_polygon):
                        obstacle_on_braille_block = True
                        break
                if obstacle_on_braille_block:
                    break

            result_queue.put((frame, obstacle_on_braille_block))

            # 마지막 감지 시간을 현재 시간으로 갱신
            last_detection_time = current_time
        else:
            # 감지 시간 이외에는 기존 프레임을 그대로 전달
            result_queue.put((frame, None))

def display_frames():
    """큐에서 결과를 가져와 화면에 표시하는 함수"""
    while True:
        item = result_queue.get()
        if item is None:
            break
        frame, obstacle_on_braille_block = item
        if obstacle_on_braille_block is not None:
            if obstacle_on_braille_block:
                print("Obstacle is present on the braille block.")
            else:
                print("No obstacle on the braille block.")
        cv2.imshow('Video Stream', frame)
        
        # 프레임 속도 제어: FPS에 맞게 지연 시간을 유지
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  # 모든 창 닫기

# 스레드 시작: 하나는 프레임을 캡처, 다른 하나는 추론을 수행, 마지막은 결과를 디스플레이
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_frames)

capture_thread.start()
process_thread.start()
display_thread.start()

capture_thread.join()
process_thread.join()
display_thread.join()

# 동영상 객체 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
