import cv2

# 加载大尺寸图片
image_path = "/Users/panding/workspace/ws_3/data/1/co_1.bmp"
original_image = cv2.imread(image_path)

# 缩放图像以适应屏幕大小
scale_percent = 50  # 设置缩放比例，这里是50%
width = int(original_image.shape[1] * scale_percent / 100)
height = int(original_image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)

# 定义回调函数获取鼠标点击位置
clicked_point = None
def get_clicked_point(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# 在窗口上设置鼠标点击事件的回调函数
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_clicked_point)

# 显示图像并等待鼠标点击
while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # 按下 'q' 键退出循环
        break
    if clicked_point is not None:
        # 显示点击位置
        print("Clicked point coordinates (scaled):", clicked_point)
        # 计算原始图像上的点击位置
        original_clicked_point = (int(clicked_point[0] / (scale_percent / 100)), int(clicked_point[1] / (scale_percent / 100)))
        print("Clicked point coordinates (original):", original_clicked_point)
        clicked_point = None  # 重置点击位置

# 关闭窗口
cv2.destroyAllWindows()