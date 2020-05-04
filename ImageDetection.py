import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

import numpy as np

#%config InlineBackend.figure_format = 'svg'

options = {
    'model' : 'C:\\Users\\LeeHyungGeol\\PycharmProjects\\ImageDetection\\venv\\Lib\\site-packages\\darkflow\\cfg\\my-tiny-yolo.cfg',
    'load' : 90530,
    'threshold' : 0.01
#    'batch' : 1
#    'gpu' : 1.0
}

tfnet = TFNet(options)
tfnet.load_from_ckpt()
img = cv2.imread('C:\\data\\testset\\Dwaejigukbap_937.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
print(result)
