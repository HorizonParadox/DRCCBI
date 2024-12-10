import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


directory_1_1 = '../../images/OCR/test_images/1_1/'
directory_1_2 = '../../images/OCR/test_images/1_2/'

directory_3_1 = '../../images/OCR/test_images/3_1/not_scan/'
directory_3_2 = '../../images/OCR/test_images/3_2/'

directory_5_1 = '../../images/OCR/test_images/5_1/'
directory_5_2 = '../../images/OCR/test_images/5_2/'

directory_7_1 = '../../images/OCR/test_images/7_1/'
directory_7_2 = '../../images/OCR/test_images/7_2/'

directory_10_1 = '../../images/OCR/test_images/10_1/'
directory_10_2 = '../../images/OCR/test_images/10_2/'

directory_13_1 = '../../images/OCR/test_images/13_1/'

directory_28_1 = '../../images/OCR/test_images/28_1/'
directory_28_2 = '../../images/OCR/test_images/28_2/'

directory_46_1 = '../../images/OCR/test_images/46_1/'
directory_46_2 = '../../images/OCR/test_images/46_2/'

my_image = cv.imread(directory_46_2 + '46_2.jpg')
scan_image = cv.imread(directory_46_2 + '46_2_scan.png')
orig_image = cv.imread(directory_46_2 + '46_2_orig.jpg')
any_scan_image = cv.imread(directory_46_2 + '46_2_any_scanner.jpg')
cam_scanner_image = cv.imread(directory_46_2 + '46_2_cam_scanner.jpg')
tap_scanner_image = cv.imread(directory_46_2 + '46_2_tap_scanner.jpg')

height_image, width_image, _ = scan_image.shape
scan_gray = cv.cvtColor(scan_image, cv.COLOR_BGR2GRAY)

my_image = cv.resize(my_image, (width_image, height_image))
orig_image = cv.resize(orig_image, (width_image, height_image))
any_scan_image = cv.resize(any_scan_image, (width_image, height_image))
cam_scanner_image = cv.resize(cam_scanner_image, (width_image, height_image))
tap_scanner_image = cv.resize(tap_scanner_image, (width_image, height_image))

print(f'SSIM for my image: {ssim(scan_gray, cv.cvtColor(my_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for orig image: {ssim(scan_gray, cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Any scan: {ssim(scan_gray, cv.cvtColor(any_scan_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Cam scanner: {ssim(scan_gray, cv.cvtColor(cam_scanner_image, cv.COLOR_BGR2GRAY))}')
print(f'SSIM for Tap scanner: {ssim(scan_gray, cv.cvtColor(tap_scanner_image, cv.COLOR_BGR2GRAY))}')

print('\n')

print(f'MSE for my image: {mse(scan_image, my_image)}')
print(f'MSE for orig image: {mse(scan_image, orig_image)}')
print(f'MSE for Any scan: {mse(scan_image, any_scan_image)}')
print(f'MSE for Cam scanner: {mse(scan_image, cam_scanner_image)}')
print(f'MSE for Tap scanner: {mse(scan_image, tap_scanner_image)}')

print('\n')

print(f'NRMSE for my image: {nrmse(scan_image, my_image)}')
print(f'NRMSE for orig image: {nrmse(scan_image, orig_image)}')
print(f'NRMSE for Any scan: {nrmse(scan_image, any_scan_image)}')
print(f'NRMSE for Cam scanner: {nrmse(scan_image, cam_scanner_image)}')
print(f'NRMSE for Tap scanner: {nrmse(scan_image, tap_scanner_image)}')

