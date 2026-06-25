import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import cv2
def compute_fuliye_spectrum(image_path):
    # 加载图像（灰度）
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 避免 log(0)

    # clip 和 normalize 到 0~255 范围
    magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255)

    return magnitude_spectrum

def get_fuliye_image_rgb(magnitude_spectrum):
    # 可视化为彩色图
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(magnitude_spectrum, cmap='viridis')
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    image_rgb = Image.open(buf).convert("RGB")
    return image_rgb