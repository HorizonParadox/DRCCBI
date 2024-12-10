import os
import cv2
from matplotlib import pyplot as plt
from src.main.document_recovery import DocumentRecovery

NUM_OF_GRID_LINE = 9
POLYNOMIAL_DEGREE = 3
NUM_POINT_PERCENT = 0.1
THRESHOLD_COEFFICIENT = 0.05
INPUT_DIRECTORY = "../../images/new_med_docs"
INPUT_DIRECTORY_DEPTH = "../../document_depth_maps"
OUTPUT_DIRECTORY = "../../images/output"
WEIGHT_DIRECTORY = '../../weights/best.pt'

for image_filename in os.listdir(INPUT_DIRECTORY):
    image_file = os.path.join(INPUT_DIRECTORY, image_filename)
    if os.path.isfile(image_file):
        print(image_filename)

        model = DocumentRecovery(image_file, WEIGHT_DIRECTORY)
        height_image, width_image, _ = model.initial_image.shape
        num_points = int(width_image * NUM_POINT_PERCENT)
        print(f'num_points: {num_points}')

        num_of_mask = 0
        # Функция Get mask и поиск контура
        polynomial = model.find_polynomial(thres_coef=THRESHOLD_COEFFICIENT)
        for concave, conv, cont in polynomial:
            num_of_mask += 1

            # Построение контура
            model.plot_polynomial(concave, conv, cont)

            # Поиск сторон в контуре
            left, right, top, bottom = model.find_edges(cont, conv)

            # Получение приблизительной длины и ширины контура
            outline_length, outline_width = model.get_dimensions(left, right, top, bottom)

            # Отображение полученных сторон разным цветом
            model.plot_colorful_edges(left, right, top, bottom)

            # Интерполяция двух кривых
            new_left_x, new_left_y, new_right_x, new_right_y = model.calculate_opposite_interpolate(left, right, True)
            new_bottom_x, new_bottom_y, new_top_x, new_top_y = model.calculate_opposite_interpolate(bottom, top, True)

            # Построение линий аппроксимации (сетка)
            linfit_x_bt, linfit_y_bt, linfit_x_lr, linfit_y_lr = model.get_lines_of_approximation(
                NUM_OF_GRID_LINE, new_left_x, new_left_y, new_right_x, new_right_y, new_bottom_x, new_bottom_y,
                new_top_x, new_top_y)

            plt.imshow(model.initial_image)
            # Построение интерполяции аппроксимации (горизонт, вертикал)
            polys_bt = model.get_interpolation_approximation(
                width_image, NUM_OF_GRID_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_bt, linfit_y_bt)
            polys_lr = model.get_interpolation_approximation(
                height_image, NUM_OF_GRID_LINE, POLYNOMIAL_DEGREE, num_points, linfit_x_lr, linfit_y_lr)
            plt.show()

            #  Поиск пересечений между линиями, заданными интерполяционными аппроксимациями
            points_set = model.get_intersection_interpolation_approximation(
                outline_length, outline_width, polys_bt, polys_lr)

            # Получение координат исправленной сетки
            XYpairs = model.get_fix_grid_coordinates(outline_length, outline_width, NUM_OF_GRID_LINE)

            # Восстановление искажения перспективы изображения
            crop_warped_image = model.remap_image(height_image, width_image, points_set, XYpairs)

            # Превращает документ в чёрно-белый скан
            # scan = DC.get_black_white_scan(crop_warped_image)

            image_name, image_extension = image_filename.split('.')
            cv2.imwrite(OUTPUT_DIRECTORY + f'/{image_name}({num_of_mask}).{image_extension}', crop_warped_image)
