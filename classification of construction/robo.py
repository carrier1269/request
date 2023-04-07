from roboflow import Roboflow
from pathlib import Path
import cv2
import base64
img = cv2.imread("prediction.jpg")
cv2.imshow("",img)
cv2.waitKey(0)

# rf = Roboflow(api_key="WrFcgRAE0jnhmUqmM7nB")
# project = rf.workspace().project("crack-dbrlf")
# model = project.version(1).model

# model.predict("crack_edge/mask_opencv big (1).jpg", confidence=40, overlap=30).save("prediction.jpg")
# cv2.waitKey(0)
# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())