from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import cv2
import numpy as np
from django.conf import settings

# Load YOLOv8 model (sửa đường dẫn đến file model của bạn)
model = YOLO('../yolov8n-face.pt')

def detect_objects(frame):
    """
    Nhận diện đối tượng trong một frame video/ảnh.
    """
    # Dự đoán
    results = model(frame)
    
    # Lấy bounding boxes và nhãn
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = (
                int(box.xyxy[0]), int(box.xyxy[1]),
                int(box.xyxy[2]), int(box.xyxy[3]),
                float(box.conf[0]), int(box.cls[0])
            )
            label = model.names[cls]
            detections.append({
                "label": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
            })
    return detections

@csrf_exempt
def yolo_detect(request):
    """
    API nhận ảnh từ client và trả về kết quả nhận diện.
    """
    if request.method == "POST" and request.FILES.get('image'):
        # Đọc file ảnh từ request
        file = request.FILES['image']
        np_image = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Nhận diện
        detections = detect_objects(frame)
        
        return JsonResponse({"detections": detections})
    return JsonResponse({"error": "Invalid request"}, status=400)

def gen_frames():
    """
    Trả về từng frame có kết quả nhận diện YOLO.
    """
    cap = cv2.VideoCapture(0)  # Sử dụng camera (ID=0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Nhận diện đối tượng
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@csrf_exempt
def video_feed(request):
    """
    Trả về luồng video với nhận diện YOLO.
    """
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
