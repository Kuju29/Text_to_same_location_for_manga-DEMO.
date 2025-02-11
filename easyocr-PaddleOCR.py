import cv2, requests, easyocr, os, sys
import numpy as np
import matplotlib.pyplot as plt

from pythainlp.tokenize import word_tokenize
from PIL import ImageFont, ImageDraw, Image

from paddleocr import PaddleOCR, draw_ocr

def rect_from_bbox(coord):
    xs = [pt[0] for pt in coord]
    ys = [pt[1] for pt in coord]
    return min(xs), min(ys), max(xs), max(ys)

def is_close_or_overlapping(r1, r2, threshold=10):
    x_min1, y_min1, x_max1, y_max1 = r1
    x_min2, y_min2, x_max2, y_max2 = r2
    overlap_x = not (x_max1 < x_min2 or x_max2 < x_min1)
    overlap_y = not (y_max1 < y_min2 or y_max2 < y_min1)
    if overlap_x and overlap_y:
        return True

    dist_x = 0
    if x_max1 < x_min2:
        dist_x = x_min2 - x_max1
    elif x_max2 < x_min1:
        dist_x = x_min1 - x_max2

    dist_y = 0
    if y_max1 < y_min2:
        dist_y = y_min2 - y_max1
    elif y_max2 < y_min1:
        dist_y = y_min1 - y_max2

    dist = dist_x + dist_y
    return (dist <= threshold)

def find_groups(rects, threshold=10):
    n = len(rects)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if is_close_or_overlapping(rects[i], rects[j], threshold):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False]*n
    group_id = [-1]*n
    current_group = 0

    def dfs(start):
        stack = [start]
        group_id[start] = current_group
        visited[start] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    group_id[v] = current_group
                    stack.append(v)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            current_group += 1
    return group_id, current_group

def translate_text(text, target_lang="th"):
    base_url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "auto",
        "tl": target_lang,
        "dt": "t",
        "q": text
    }
    r = requests.get(base_url, params=params)
    try:
        result = r.json()
        if result and len(result) > 0 and result[0]:
            segments = result[0]
            translated_segments = []
            for seg in segments:
                if seg and len(seg) > 0:
                    translated_segments.append(seg[0])
            translated_text = " ".join(translated_segments)
            return translated_text
    except Exception as e:
        print("Translate Error:", e)
    return text

def merge_trailing_dash_tokens(tokens):
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token = token.replace(";", "").replace(":", "") 

        if token.endswith('-'):
            token_no_dash = token[:-1]
            if i + 1 < len(tokens):
                next_token = tokens[i+1]
                next_token = next_token.replace(";", "").replace(":", "")
                
                combined = token_no_dash + next_token
                new_tokens.append(combined)
                i += 2
            else:
                new_tokens.append(token_no_dash)
                i += 1
        else:
            new_tokens.append(token)
            i += 1
    return new_tokens

def wrap_text_thai(text, font, box_w, line_spacing=4):
    pil_img = Image.new("RGB", (1,1), (255,255,255))
    draw = ImageDraw.Draw(pil_img)
    tokens = word_tokenize(text, engine="newmm")

    lines = []
    current_line = ""

    for token in tokens:
        test_line = current_line + token
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        line_width = right - left

        if line_width > box_w and current_line != "":
            lines.append(current_line)
            current_line = token
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)
    return lines

def measure_multiline(lines, font, line_spacing=4):
    pil_img = Image.new("RGB", (1,1), (255,255,255))
    draw = ImageDraw.Draw(pil_img)

    max_width = 0
    total_height = 0
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        w = right - left
        h = bottom - top
        max_width = max(max_width, w)
        total_height += (h + line_spacing)
    total_height -= line_spacing
    return max_width, total_height

def find_best_font_size(text, font_path, min_size, max_size, box_w, box_h, line_spacing=4):
    best_size = min_size
    best_lines = [text]

    low = min_size
    high = max_size

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)
        lines = wrap_text_thai(text, font, box_w, line_spacing=line_spacing)
        w, h = measure_multiline(lines, font, line_spacing=line_spacing)
        if w <= box_w and h <= box_h:
            best_size = mid
            best_lines = lines
            low = mid + 1
        else:
            high = mid - 1

    return best_size, best_lines

def draw_multiline_center(img, lines, font_path, font_size, box, color=(255,0,0), line_spacing=4):
    x_min, y_min, x_max, y_max = box
    box_w = x_max - x_min
    box_h = y_max - y_min

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    max_width, total_height = measure_multiline(lines, font, line_spacing=line_spacing)

    current_y = y_min + (box_h - total_height) / 2

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        line_w = right - left
        line_h = bottom - top
        x_start = x_min + (box_w - line_w) / 2 
        draw.text((x_start, current_y), line, font=font, fill=color)
        current_y += (line_h + line_spacing)

    return np.array(pil_img)

if __name__ == "__main__":
    font_path = "THSarabunNew.ttf"
    if not os.path.exists(font_path):
        print("ไม่พบไฟล์ฟอนต์", font_path)
        sys.exit()

    TARGET_LANG = "th"

    reader_easy = easyocr.Reader(['en'], gpu=False)

    reader_paddle = PaddleOCR(use_angle_cls=True, lang='en')

    image_path = "2.jpg"
    if not os.path.exists(image_path):
        print(f"❌ ไม่พบไฟล์ '{image_path}'")
        sys.exit()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ OpenCV ไม่สามารถโหลดรูปได้")
        sys.exit()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    easy_results = reader_easy.readtext(img_rgb, detail=1)

    ocr_data = []
    for (coord, text, conf) in easy_results:
        x_min, y_min, x_max, y_max = rect_from_bbox(coord)
        ocr_data.append({
            "bbox": (x_min, y_min, x_max, y_max),
            "confidence": conf
        })

    rects = [d["bbox"] for d in ocr_data]
    group_id, total_groups = find_groups(rects, threshold=10)

    group_data = {}
    for g in range(total_groups):
        group_data[g] = {
            "group_id": g,
            "x_min": float('inf'),
            "y_min": float('inf'),
            "x_max": float('-inf'),
            "y_max": float('-inf')
        }

    for i, item in enumerate(ocr_data):
        g_id = group_id[i]
        (x_min, y_min, x_max, y_max) = item["bbox"]
        group_data[g_id]['x_min'] = min(group_data[g_id]['x_min'], x_min)
        group_data[g_id]['y_min'] = min(group_data[g_id]['y_min'], y_min)
        group_data[g_id]['x_max'] = max(group_data[g_id]['x_max'], x_max)
        group_data[g_id]['y_max'] = max(group_data[g_id]['y_max'], y_max)

    img_result = img_rgb.copy()

    for g_id in range(total_groups):
        x_min = int(group_data[g_id]['x_min'])
        y_min = int(group_data[g_id]['y_min'])
        x_max = int(group_data[g_id]['x_max'])
        y_max = int(group_data[g_id]['y_max'])

        cv2.rectangle(img_result, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

        sub_img = img_rgb[y_min:y_max, x_min:x_max]
        paddle_result = reader_paddle.ocr(sub_img)

        confidence_threshold = 0.60

        recognized_texts = []
        for line in paddle_result:
            if not line:
                continue
            for word_info in line:
                text_part = word_info[1][0]
                conf = word_info[1][1]
                
                if conf >= confidence_threshold:
                    recognized_texts.append(text_part)
                else:
                    print(f"❌ ข้าม '{text_part}' เนื่องจากความมั่นใจต่ำ: {conf:.2f}")

        merged_tokens = merge_trailing_dash_tokens(recognized_texts)
        merged_text = " ".join(merged_tokens)
        print(f"[Group {g_id}] PaddleOCR text:", merged_text)

        translated_text = translate_text(merged_text, TARGET_LANG)
        print(f" -> Translated ({TARGET_LANG}):", translated_text)

        margin = 10
        box_w = (x_max - x_min) - 2 * margin
        box_h = (y_max - y_min) - 2 * margin

        if box_w < 10 or box_h < 10:
            continue

        best_font_size, lines = find_best_font_size(
            text=translated_text,
            font_path=font_path,
            min_size=10,
            max_size=60,
            box_w=box_w,
            box_h=box_h,
            line_spacing=4
        )

        inner_box = (x_min + margin, y_min + margin, x_max - margin, y_max - margin)
        img_result = draw_multiline_center(
            img=img_result,
            lines=lines,
            font_path=font_path,
            font_size=best_font_size,
            box=inner_box,
            color=(255,0,0),
            line_spacing=4
        )

    img_result_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_result)
    plt.axis("off")
    plt.title("Remove Original Text using EasyOCR BBox + Translated Text (PaddleOCR)")
    plt.show()
