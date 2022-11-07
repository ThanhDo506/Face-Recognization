import keras
import cv2
import numpy as np
import os

# # đường dẫn là folder chứa ảnh thư mục tất cả người để lưu nhãn
mypath = 'Preprocessing/resource/raw'
labels = os.listdir(mypath)
print(labels)
# Sau thi thực hiện lệnh sẽ được list như sau
# labels = ['DaiNV','DungNV','HienNV','HoangAnh','HoangNH','HoanND','HungDV','LuyenPQ','MaiNT','MaiPT','MinhNT','NghiNV','PhuongNP','SyKD','TaiDD','TaiNV','Thinh','Thuy','TruongNV','YenBT']
class Recognition:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.model = keras.models.load_model(modelPath, compile=False)
    
    def predict(self, imagePath):
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img * 1.0 / 255
        img = np.array([img])
        res = self.model.predict(img)[0]
        return labels[np.argmax(res)]