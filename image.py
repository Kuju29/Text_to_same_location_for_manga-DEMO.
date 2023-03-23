from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision, translate_v2 as translate
from google.oauth2 import service_account
import os

image_path = "new1.png"
translate_text = "th"
spacing = 5
padding = 5 
row_threshold = 10
service_account_file = "service_account.json"
font_path = r"C:\Windows\Fonts\angsana.ttc"
translation_image = True

image_basename = os.path.splitext(os.path.basename(image_path))[0]
image = Image.open(image_path)
image_copy = image.copy()

credentials = service_account.Credentials.from_service_account_file(
    service_account_file)
translate_client = translate.Client(
    credentials=credentials)
client = vision.ImageAnnotatorClient(
    credentials=credentials)

def detect_text(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    return texts

# def detect_text(image_path):
#     with open(image_path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)
#     response = client.document_text_detection(image=image)
#     texts = response.text_annotations

#     return texts

data = detect_text(image_path)

credentials = service_account.Credentials.from_service_account_file(
    service_account_file)
translate_client = translate.Client(
    credentials=credentials)

font = ImageFont.truetype(font_path, 20)

whitedraw = ImageDraw.Draw(image_copy)

for text in data[1:]:
    vertices = text.bounding_poly.vertices
    x, y, w, h = vertices[0].x, vertices[0].y, vertices[2].x - vertices[0].x, vertices[2].y - vertices[0].y
    whitedraw.rectangle([x - padding, y - padding, x + w + padding, y + h + padding], fill=(255, 255, 255))

output_image_path = f"whitedraw-{image_basename}.png"
image_copy.save(output_image_path)

white_rectangle_image = Image.open(output_image_path)
draw = ImageDraw.Draw(white_rectangle_image)

prev_x = None
prev_y = None

for text in data[1:]:
    vertices = text.bounding_poly.vertices
    x, y, w, h = vertices[0].x, vertices[0].y, vertices[2].x - vertices[0].x, vertices[2].y - vertices[0].y
    word = text.description
    if translation_image:
        translation = translate_client.translate(word, target_language=translate_text)
        text = translation["translatedText"]
    else:
        text = word


    text_w = draw.textlength(text, font=font)


    if prev_x is not None and prev_y is not None:

        if abs(prev_y - y) < row_threshold:
            x = prev_x + spacing
        else:
            prev_x = x

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    prev_x = x + text_w
    prev_y = y



final_output_image_path = f"output-{image_basename}.png"
white_rectangle_image.save(final_output_image_path)

image.close()
image_copy.close()
white_rectangle_image.close()
os.remove(output_image_path)
