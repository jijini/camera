from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import cv2

# 모델 불러오기
model = keras.models.load_model("D:\python\ex230419\keras_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 클래스 이름 목록
class_names = ["HAECHAN", "JAEMIN", "CHENLE", "MARK"]

# 웹캠에서 이미지 캡처
cap = cv2.VideoCapture(0)

while True:
    # 캡처된 이미지를 PIL.Image 객체로 변환
    ret, frame = cap.read()
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 이미지 전처리 과정
    # 웹캠에서 이미지 캡처
 # 이미지 전처리 과정
    size = (224, 224)
    pil_image = ImageOps.fit(pil_image, size, Image.LANCZOS)
    image_array = np.array(pil_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = normalized_image_array.reshape((1, 224, 224, 3))

    # 모델 예측
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 결과 출력
    result_text = f"Class: {class_name}, Confidence Score: {confidence_score:.2%}"
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Capturing', frame)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows() 


    
