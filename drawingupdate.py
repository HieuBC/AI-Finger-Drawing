import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)
# Tăng kích thước cửa sổ để có nhiều không gian vẽ hơn (Tuỳ chọn)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Lấy kích thước khung hình thực tế
ret, frame = cap.read()
h_frame, w_frame, _ = frame.shape
canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)

# --- BIẾN TRẠNG THÁI ---
prev_x, prev_y = 0, 0
drawing_mode = False  
fingers_touching = False  

# Mặc định: Nét vẽ màu Đỏ (BGR), Kích thước 5
draw_color = (0, 0, 255) 
brush_size = 5

# --- CẤU HÌNH GIAO DIỆN (UI) ---
# Bảng màu (Color Palette) - Danh sách mã màu BGR
colors = [
    (0, 0, 255),    # Đỏ
    (0, 255, 255),  # Vàng
    (0, 255, 0),    # Xanh lá
    (255, 0, 0),    # Xanh dương
    (255, 0, 255),  # Hồng / Magenta
    (255, 255, 255) # Trắng
]
color_radius = 20
color_centers = [] # Lưu tọa độ tâm các nút màu

# Nút chỉnh kích thước (Brush Size)
sizes = [5, 10, 20] # Nhỏ, Vừa, Lớn
size_centers = [] # Lưu tọa độ tâm các nút kích thước

# Nút Xóa (Clear Button)
button_x1, button_y1 = w_frame // 2 - 75, 10
button_x2, button_y2 = w_frame // 2 + 75, 60

# --- CÁC HÀM HỖ TRỢ ---
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def fingers_are_open(landmarks):
    index_tip = landmarks[8]
    index_knuckle = landmarks[6]
    thumb_tip = landmarks[4]
    thumb_knuckle = landmarks[2]
    return index_tip.y < index_knuckle.y and thumb_tip.y < thumb_knuckle.y

while True:
    ret, frame = cap.read()
    if not ret: break

    # Lật khung hình như gương
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # --- VẼ GIAO DIỆN LÊN CAMERA ---
    # 1. Nút Clear ở giữa trên cùng
    cv2.rectangle(frame, (button_x1, button_y1), (button_x2, button_y2), (50, 50, 50), -1)
    cv2.putText(frame, "CLEAR", (button_x1 + 20, button_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Bảng chọn màu (Cạnh phải)
    color_centers.clear()
    for i, color in enumerate(colors):
        cx, cy = w_frame - 50, 80 + (i * 70)
        color_centers.append((cx, cy, color))
        # Vẽ viền trắng nếu đang được chọn
        thickness = -1 if color == draw_color else 3
        cv2.circle(frame, (cx, cy), color_radius, color, thickness)
        if color == draw_color:
            cv2.circle(frame, (cx, cy), color_radius + 5, (255, 255, 255), 2)

    # 3. Chỉnh kích thước nét vẽ (Cạnh trái)
    size_centers.clear()
    for i, size in enumerate(sizes):
        cx, cy = 50, 150 + (i * 100)
        size_centers.append((cx, cy, size))
        # Nút kích thước hiển thị theo màu đang chọn
        cv2.circle(frame, (cx, cy), size, draw_color, -1)
        cv2.circle(frame, (cx, cy), 30, (200, 200, 200), 2) # Viền cố định
        if size == brush_size:
            cv2.circle(frame, (cx, cy), 35, (0, 255, 0), 3) # Highlight khi chọn

    # --- XỬ LÝ NHẬN DIỆN TAY ---
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            index_x, index_y = int(index_finger_tip.x * w_frame), int(index_finger_tip.y * h_frame)
            thumb_x, thumb_y = int(thumb_tip.x * w_frame), int(thumb_tip.y * h_frame)

            # Tính khoảng cách chụm ngón tay
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)

            # --- TƯƠNG TÁC VỚI GIAO DIỆN KHI KHÔNG CHỤM TAY ---
            if distance > 50:
                # Kiểm tra chạm nút Màu
                for cx, cy, color in color_centers:
                    if calculate_distance(index_x, index_y, cx, cy) < color_radius:
                        draw_color = color
                        
                # Kiểm tra chạm nút Kích thước
                for cx, cy, size in size_centers:
                    if calculate_distance(index_x, index_y, cx, cy) < 30:
                        brush_size = size

                # Kiểm tra chạm nút Clear
                if button_x1 < index_x < button_x2 and button_y1 < index_y < button_y2:
                    canvas = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)

            # --- KÍCH HOẠT VẼ KHI CHỤM TAY ---
            if distance < 40:
                if not fingers_touching and fingers_are_open(hand_landmarks.landmark):
                    drawing_mode = not drawing_mode
                    if drawing_mode:
                        time.sleep(0.2) 
                fingers_touching = True
            else:
                fingers_touching = False

            # Thực hiện vẽ lên Canvas
            if drawing_mode:
                # Vẽ thêm hình tròn nhỏ ở đầu ngón tay để tạo cảm giác "Neon"
                cv2.circle(frame, (index_x, index_y), brush_size, draw_color, -1)
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y
                
                # Vẽ đường nối tiếp (Áp dụng màu và kích thước đã chọn)
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), draw_color, brush_size * 2)
                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = 0, 0

            # (Tuỳ chọn) Vẽ khung xương tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Trộn khung hình camera và canvas vẽ
    # Dùng addWeighted để nét vẽ có hiệu ứng sáng (Neon-like) hơn một chút
    combined_frame = cv2.addWeighted(frame, 1.0, canvas, 0.8, 0)

    cv2.imshow("AI Finger Drawing - Advanced", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()