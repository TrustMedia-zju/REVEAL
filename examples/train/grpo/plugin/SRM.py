# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# def SRM(img_path):
#     srm_filters = np.array([
#         # 第1个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, -1, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第2个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, -1, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第3个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, -1, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第4个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, -1, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0]],
#         # 第5个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, -1, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第6个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, -1, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第7个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 1, -1, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第8个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, -1, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]]
#     ])
#     image_path = img_path   # 替换为你的图像路径
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
#     # 显示原始图像
#     combined_result = np.zeros_like(image, dtype=float)

#     # 对每个滤波器应用滤波并叠加结果
#     for srm_filter in srm_filters:
#         filtered_image = cv2.filter2D(image, -1, srm_filter)
#         combined_result += filtered_image

#     # 对结果进行归一化以适应显示范围
#     combined_result1 = np.clip(combined_result, 0, 255).astype(np.uint8)



#     # plt.subplot(1, 2, 2)
#     # plt.imshow(cv2.cvtColor(combined_result, cv2.COLOR_BGR2RGB))
#     # plt.title('Combined Filter Result')
#     # plt.axis('off')

#     # plt.tight_layout()
#     # plt.savefig("combined_filter_result.png", dpi=300, bbox_inches='tight')

#     # plt.show()
#     srm_filters = np.array([
#         [[0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, -2, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第10个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, -2, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第11个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 1, -2, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第12个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, -2, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0]],
#     ])
#     image_path = img_path  # 替换为你的图像路径
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
#     # 显示原始图像
#     combined_result = np.zeros_like(image, dtype=float)

#     # 对每个滤波器应用滤波并叠加结果
#     for srm_filter in srm_filters:
#         filtered_image = cv2.filter2D(image, -1, srm_filter)
#         combined_result += filtered_image

#     # 对结果进行归一化以适应显示范围
#     combined_result2 = np.clip(combined_result, 0, 255).astype(np.uint8)

#     srm_filters = np.array([
#         [[0, 0, -1, 0, 0],
#         [0, 0, 3, 0, 0],
#         [0, 0, -3, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第14个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 3, 0],
#         [0, 0, -3, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第15个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 1, -3, 3, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第16个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, -3, 0, 0],
#         [0, 0, 0, 3, 0],
#         [0, 0, 0, 0, 0]],
#         # 第17个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [0, 0, -3, 0, 0],
#         [0, 0, 3, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第18个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, -3, 0, 0],
#         [0, 3, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第19个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 3, -3, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第20个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 3, 0, 0, 0],
#         [0, 0, -3, 0, 0],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0]],
#     ])
#     image_path = img_path   # 替换为你的图像路径
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
#     # 显示原始图像
#     combined_result = np.zeros_like(image, dtype=float)

#     # 对每个滤波器应用滤波并叠加结果
#     for srm_filter in srm_filters:
#         filtered_image = cv2.filter2D(image, -1, srm_filter)
#         combined_result += filtered_image

#     # 对结果进行归一化以适应显示范围
#     combined_result3 = np.clip(combined_result, 0, 255).astype(np.uint8)

   
#     srm_filters = np.array([
#         [[0, 0, 0, 0, 0],
#         [0, -1, 2, -1, 0],
#         [0, 2, -4, 2, 0],
#         [0, -1, 2, -1, 0],
#         [0, 0, 0, 0, 0]],
#         # 第22个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, -1, 2, -1, 0],
#         [0, 2, -4, 2, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第23个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 2, -1, 0],
#         [0, 0, -4, 2, 0],
#         [0, 0, 2, -1, 0],
#         [0, 0, 0, 0, 0]],
#         # 第24个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 2, -4, 2, 0],
#         [0, -1, 2, -1, 0],
#         [0, 0, 0, 0, 0]],
#         # 第25个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, -1, 2, 0, 0],
#         [0, 2, -4, 0, 0],
#         [0, -1, 2, 0, 0],
#         [0, 0, 0, 0, 0]],
#     ])
#     image_path = img_path   # 替换为你的图像路径
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
#     # 显示原始图像
#     combined_result = np.zeros_like(image, dtype=float)

#     # 对每个滤波器应用滤波并叠加结果
#     for srm_filter in srm_filters:
#         filtered_image = cv2.filter2D(image, -1, srm_filter)
#         combined_result += filtered_image

#     # 对结果进行归一化以适应显示范围
#     combined_result4 = np.clip(combined_result, 0, 255).astype(np.uint8)

   
#     srm_filters = np.array([
#         [[-1, 2, -2, 2, -1],
#         [2, -6, 8, -6, 2],
#         [-2, 8, -12, 8, -2],
#         [2, -6, 8, -6, 2],
#         [-1, 2, -2, 2, -1]],
#         # 第27个滤波器
#         [[-1, 2, -2, 2, -1],
#         [2, -6, 8, -6, 2],
#         [-2, 8, -12, 8, -2],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#         # 第28个滤波器
#         [[0, 0, -2, 2, -1],
#         [0, 0, 8, -6, 2],
#         [0, 0, -12, 8, -2],
#         [0, 0, 8, -6, 2],
#         [0, 0, -2, 2, -1]],
#         # 第29个滤波器
#         [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [-2, 8, -12, 8, -2],
#         [2, -6, 8, -6, 2],
#         [-1, 2, -2, 2, -1]],
#         # 第30个滤波器
#         [[-1, 2, -2, 0, 0],
#         [2, -6, 8, 0, 0],
#         [-2, 8, -12, 0, 0],
#         [2, -6, 8, 0, 0],
#         [-1, 2, -2, 0, 0]]
#     ])
#     image_path = img_path   # 替换为你的图像路径
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
#     # 显示原始图像
#     combined_result = np.zeros_like(image, dtype=float)

#     # 对每个滤波器应用滤波并叠加结果
#     for srm_filter in srm_filters:
#         filtered_image = cv2.filter2D(image, -1, srm_filter)
#         combined_result += filtered_image

#     # 对结果进行归一化以适应显示范围
#     combined_result5 = np.clip(combined_result, 0, 255).astype(np.uint8)
#     return combined_result1,combined_result2,combined_result3,combined_result4,combined_result5

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def SRM(img_path):
    srm_filters = np.array([
        # 第1个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第2个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第3个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第4个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]],
        # 第5个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第6个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第7个滤波器
        [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],
        # 第8个滤波器
        [[0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]
    ])
    image_path = img_path   # 替换为你的图像路径
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to read the image file. Please check the file path and ensure the file exists.")
    # 显示原始图像
    combined_result = np.zeros_like(image, dtype=float)

    # 对每个滤波器应用滤波并叠加结果
    for srm_filter in srm_filters:
        filtered_image = cv2.filter2D(image, -1, srm_filter)
        combined_result += filtered_image

    # 对结果进行归一化以适应显示范围
    combined_result1 = np.clip(combined_result, 0, 255).astype(np.uint8)
    return combined_result1