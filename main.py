import cv2
from selenium.webdriver import ActionChains

GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA_X = 0
CANNY_THRESHOLD1 = 200
CANNY_THRESHOLD2 = 450


def get_gaussian_blur_image(image):
    return cv2.GaussianBlur(image, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X)


def get_canny_image(image):
    return cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)


def get_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contour_area_threshold(image_width, image_height):
    contour_area_min = (image_width * 0.15) * (image_height * 0.25) * 0.8
    contour_area_max = (image_width * 0.15) * (image_height * 0.25) * 1.2
    return contour_area_min, contour_area_max


def get_arc_length_threshold(image_width, image_height):
    arc_length_min = ((image_width * 0.15) + (image_height * 0.25)) * 2 * 0.8
    arc_length_max = ((image_width * 0.15) + (image_height * 0.25)) * 2 * 1.2
    return arc_length_min, arc_length_max


def get_offset_threshold(image_width):
    offset_min = 0.2 * image_width
    offset_max = 0.85 * image_width
    return offset_min, offset_max


def main():
    # 加载一个名为“captcha.png”的图像文件
    image_raw = cv2.imread('captcha.png')
    print(image_raw.shape)
    # 将图像的高度、宽度分别赋值给变量“image_height”和“image_width”
    image_height, image_width, _ = image_raw.shape
    # 使用OpenCV库中的“cv2.GaussianBlur”函数对原始图像进行高斯模糊处理，返回处理后的图像
    image_gaussian_blur = get_gaussian_blur_image(image_raw)
    # 使用OpenCV库中的“cv2.Canny”函数对高斯模糊处理后的图像进行边缘检测，返回处理后的图像
    '''
    边缘检测是一种计算机视觉技术，用于检测图像中的物体边缘。Canny边缘检测算法是最常用的边缘检测算法之一。该算法的基本思想是，
    先用高斯滤波器对图像进行平滑处理，然后计算图像的梯度，找到梯度最大的位置作为边缘点，并用非极大值抑制和双阈值处理来增强边缘
    '''
    image_canny = get_canny_image(image_gaussian_blur)
    # OpenCV库中的“cv2.findContours”函数对边缘检测后的图像进行轮廓检测，返回一个轮廓的列表，其中每个轮廓由一系列点的坐标组成。
    contours = get_contours(image_canny)
    # print(contours)
    # 保存了经过Canny边缘检测处理后的image_canny
    cv2.imwrite('image_canny.png', image_canny)
    # 保存了经过高斯模糊处理后的image_gaussian_blur。这样做是为了方便后续的调试和分析，可以在不同的阶段查看处理后的图像效果。
    cv2.imwrite('image_gaussian_blur.png', image_gaussian_blur)
    # 计算出轮廓的最小和最大面积，然后将这两个值作为元组的形式返回。在后面的代码中，这两个值会被用来过滤掉面积过小或过大的轮廓。
    contour_area_min, contour_area_max = get_contour_area_threshold(image_width, image_height)
    # 该函数的作用是计算出一个阈值范围，用于筛选轮廓。具体来说，该函数根据输入的image_width和image_height，计算出轮廓的最小和最大周长，然后将这两个值作为元组的形式返回。在后面的代码中，这两个值会被用来过滤掉周长过小或过大的轮廓。
    arc_length_min, arc_length_max = get_arc_length_threshold(image_width, image_height)
    # 该函数根据输入的image_width，计算出偏移量的最小和最大值，然后将这两个值作为元组的形式返回。在后面的代码中，这两个值会被用来过滤掉位置在太左或太右的轮廓。
    offset_min, offset_max = get_offset_threshold(image_width)
    offset = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if contour_area_min < cv2.contourArea(contour) < contour_area_max and \
                arc_length_min < cv2.arcLength(contour, True) < arc_length_max and \
                offset_min < x < offset_max:
            cv2.rectangle(image_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)
            offset = x
    cv2.imwrite('image_label.png', image_raw)
    print('offset', offset)

    # 将滑块移动指定距离
    if offset:
        slider = get_slider(driver)
        action = ActionChains(driver)
        action.click_and_hold(slider).perform()
        action.move_by_offset(offset, 0).perform()
        action.release().perform()


if __name__ == '__main__':
    main()

