import tensorflow as tf
import cv2
import numpy as np
import os
from urllib.request import urlopen


# đường dẫn là folder chứa ảnh thư mục tất cả người để lưu nhãn
mypath = 'C:\\Users\\wwwng\\Desktop\\Haizzzz\\Face Recognization\\lumi_face\\val'
labels = os.listdir(mypath)
# Sau thi thực hiện lệnh sẽ được list như sau
# labels = ['DaiNV','DungNV','HienNV','HoangAnh','HoangNH','HoanND','HungDV','LuyenPQ','MaiNT','MaiPT','MinhNT','NghiNV','PhuongNP','SyKD','TaiDD','TaiNV','Thinh','Thuy','TruongNV','YenBT']
class Recognition:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.model = tf.keras.models.load_model(modelPath, compile=False)
    
    def predict(self, imagePath):
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img * 1.0 / 255
        img = np.array([img])
        res = self.model.predict(img)[0]
        for i in range(len(labels)):
            if res[i] == 1:
                break
        return labels[i]