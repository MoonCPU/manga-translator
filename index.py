import os
import time

import cv2
from cv2 import dnn_superres
import pytesseract
from PIL import Image

folder_path = "D:/Biblioteca-Moon/screenshots"
file_path = 'D:/Biblioteca-Moon/screenshots/image.png'

# Load input image
newImage = Image.open(file_path)

# Resize input image
resizedImage = newImage.resize((500, 500))
resizedImage.save(file_path)

# Load Super Resolution model
sr = dnn_superres.DnnSuperResImpl_create()
path = "./EDSR_x2.pb"
sr.readModel(path)
sr.setModel("edsr", 2)

# Load resized image and upscale with Super Resolution model
image = cv2.imread(file_path)
upscaled = sr.upsample(image)

# Save upscaled image
cv2.imwrite("D:/Biblioteca-Moon/screenshots/image.png", upscaled)

# Perform OCR on upscaled image
text = pytesseract.image_to_string(upscaled, lang="jpn_vert")
print(text)


# while True:
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png"):
#             # Perform OCR on the image
#             # ...
            
            
#             # Remove the image from the folder
#             os.remove(os.path.join(folder_path, filename))
#     time.sleep(1)