import cv2, time
import numpy as np
import matplotlib.pyplot as plt

# โมดูลสำหรับ OCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from paddleocr import PaddleOCR
import easyocr

# ฟังก์ชันตรวจสอบว่าจุด (x, y) อยู่ภายใน bounding box หรือไม่
def point_in_box(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return (x >= x_min) and (x <= x_max) and (y >= y_min) and (y <= y_max)

# ฟังก์ชันบันทึก log
def log_message(message):
    print(message)  # สามารถเปลี่ยนเป็นบันทึกลงไฟล์ได้ เช่น open("log.txt", "a").write(message + "\n")

# ---------------------------
# 1. โหลดภาพ
# ---------------------------
image_path = "2.jpg"  # เปลี่ยน path ให้ตรงกับภาพที่ใช้งาน
image = cv2.imread(image_path)
h, w = image.shape[:2]

# รายการเก็บผลลัพธ์จากแต่ละโมเดล
detections = []

log_message("=== เริ่มกระบวนการ OCR ===")

# ---------------------------
# 2. ทำ OCR ทีละโมเดล
# ---------------------------

# (ก) OCR ด้วย EasyOCR
log_message("[EasyOCR] กำลังประมวลผล OCR...")
reader_easy = easyocr.Reader(['en'])
result_easy = reader_easy.readtext(image_path)
for det in result_easy:
    bbox_easy, text, conf = det
    x_coords = [pt[0] for pt in bbox_easy]
    y_coords = [pt[1] for pt in bbox_easy]
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    detections.append({
        "text": text,
        "bbox": (x_min, y_min, x_max, y_max),
        "center": center,
        "confidence": conf,
        "origin": "easyocr"
    })

log_message(f"[EasyOCR] ตรวจพบ {len(detections)} กล่องข้อความ")

time.sleep(5)

# (ข) OCR ด้วย docTR (ocr_predictor)
log_message("[docTR] กำลังประมวลผล OCR...")
model_docTR = ocr_predictor(pretrained=True)
doc = DocumentFile.from_images(image_path)
result_docTR = model_docTR(doc)
output_docTR = result_docTR.export()

for page in output_docTR["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                text = word["value"]
                bbox_norm = word["geometry"]
                x_min = int(bbox_norm[0][0] * w)
                y_min = int(bbox_norm[0][1] * h)
                x_max = int(bbox_norm[1][0] * w)
                y_max = int(bbox_norm[1][1] * h)
                center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                conf = word.get("confidence", 1.0)
                detections.append({
                    "text": text,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "center": center,
                    "confidence": conf,
                    "origin": "docTR"
                })

log_message(f"[docTR] ตรวจพบ {len(detections)} กล่องข้อความ")

time.sleep(5)

# (ค) OCR ด้วย PaddleOCR
log_message("[PaddleOCR] กำลังประมวลผล OCR...")
ocr_paddle = PaddleOCR(use_angle_cls=True)
result_paddle = ocr_paddle.ocr(image_path, cls=True)
for line in result_paddle:
    for word in line:
        coords = word[0]
        x_coords = [pt[0] for pt in coords]
        y_coords = [pt[1] for pt in coords]
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        text = word[1][0]
        conf = word[1][1]
        detections.append({
            "text": text,
            "bbox": (x_min, y_min, x_max, y_max),
            "center": center,
            "confidence": conf,
            "origin": "PaddleOCR"
        })

log_message(f"[PaddleOCR] ตรวจพบ {len(detections)} กล่องข้อความ")

time.sleep(5)

# ---------------------------
# 3. คัดเลือกกล่องที่ดีที่สุดตาม confidence
# ---------------------------
detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)

final_detections = []
hidden_detections = []  # เก็บรายการที่ถูกซ่อน

log_message("=== เริ่มการคัดเลือกข้อความที่จะแสดง ===")
for det in detections_sorted:
    overlapped = False
    for sel in final_detections:
        if point_in_box(det["center"], sel["bbox"]):
            overlapped = True
            hidden_detections.append(det)
            log_message(f"[ซ่อน] \"{det['text']}\" ({det['confidence']*100:.1f}%) เพราะอยู่ในพื้นที่ของ \"{sel['text']}\" ({sel['confidence']*100:.1f}%)")
            break
    if not overlapped:
        final_detections.append(det)
        log_message(f"[แสดง] \"{det['text']}\" ({det['confidence']*100:.1f}%) จาก {det['origin']}")

# ---------------------------
# 4. แสดงผลลัพธ์
# ---------------------------
log_message(f"=== สรุปผลลัพธ์ ===")
log_message(f"กล่องข้อความที่แสดง: {len(final_detections)}")
log_message(f"กล่องข้อความที่ถูกซ่อน: {len(hidden_detections)}")

for det in final_detections:
    x_min, y_min, x_max, y_max = det["bbox"]
    if det["origin"] == "docTR":
        color = (0, 255, 0)
    elif det["origin"] == "PaddleOCR":
        color = (255, 0, 0)
    elif det["origin"] == "easyocr":
        color = (0, 0, 255)
    text_with_conf = f"{det['text']} {det['confidence']*100:.1f}% ({det['origin']})"
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, text_with_conf, (x_min, max(y_min - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
