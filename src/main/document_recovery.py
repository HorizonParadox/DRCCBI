import numpy as np
import cv2
from math import ceil
import shapely.geometry as sg
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, griddata
from scipy import spatial
import itertools
from yolo_segmentation import YOLOSegmentation
from pylab import *


# 1.9.1

class DocumentRecovery:

    def __init__(self, image_path, weight_path):
        self.init_image = cv2.imread(image_path)
        self.initial_image = cv2.cvtColor(self.init_image, cv2.COLOR_BGR2RGB)
        self.weight_path = weight_path

    @staticmethod
    def find_mask_intersection(seg1, seg2):
        """
        Проверяет пересечение двух масок, представленных в виде сегментов.

        Аргументы:
        - seg1 (list): Сегмент первой маски, содержащий координаты точек.
        - seg2 (list): Сегмент второй маски, содержащий координаты точек.

        Возвращает:
        - bool: True, если маски пересекаются, и False в противном случае.
        """

        polygon1 = sg.Polygon(seg1)
        polygon2 = sg.Polygon(seg2)

        return polygon1.intersects(polygon2)

    @staticmethod
    def plot_image(image):
        """
        Выводит изображение на экран.

        Аргументы:
        - image: Массив с данными изображения.
        """

        plt.imshow(image)
        plt.show()

    def find_document_masks_yolo(self):
        """
        Находит маски документа с использованием модели YOLO.

        Возвращает:
        - masks: Список с найденными масками документа.
        """

        self.plot_image(self.initial_image)
        model = YOLOSegmentation(self.weight_path)
        bboxes, segments, scores = model.detect(self.init_image)
        print(scores)

        masks = []
        yolo_result = []
        yolo_output = []
        single_image_flag = False
        segments_intersection_flag = False

        if len(segments) == 1:
            # Если на изображении только один объект, устанавливает флаг одного изображения
            single_image_flag = True
            yolo_output = [[bboxes[0], segments[0], scores[0]]]
        elif len(segments) > 1:
            # Еслиа на изображении несколько объектов...
            single_image_flag = False

            # Определяет два объекта с наибольшими оценками
            for bbox, seg, score in zip(bboxes, segments, scores):
                yolo_result.append([bbox, seg, score])
            max1 = max(yolo_result, key=lambda x: x[2])
            yolo_result.remove(max1)
            max2 = max(yolo_result, key=lambda x: x[2])

            # Проверяет пересечение между сегментами двух объектов
            segments_intersection_flag = self.find_mask_intersection(max1[1], max2[1])

            if segments_intersection_flag:
                # Если есть пересечение, добавляет два объекта в yolo_output
                yolo_output = [max1, max2]
            else:
                # Если пересечения нет, устанавливает флаг одного изображения и добавляет объект с наибольшей оценкой
                single_image_flag = True
                yolo_output = [max([max1, max2], key=lambda x: x[2])]

        print(f'SHAPE: {self.initial_image.shape}')

        # Обрабатывает каждый объект в yolo_output
        for bbox, seg, score in yolo_output:
            if single_image_flag or segments_intersection_flag:
                print(f'Score: {score}')

                # Создает маску объекта
                mask = np.zeros(self.initial_image.shape[:2], np.uint8)
                cv2.drawContours(mask, [seg], -1, (255, 255, 255), -1, cv2.LINE_AA)
                self.plot_image(mask)

                # Определяет параметры фильтрации маски
                radius = int(min(self.initial_image.shape[0], self.initial_image.shape[1]) * 0.01)
                if radius < 2:
                    radius = 3
                epsilon = radius // 2
                print(f'Radius: {radius}')

                # Применяет фильтрацию маски
                fil_mask = cv2.ximgproc.guidedFilter(self.initial_image, mask, radius, epsilon)
                _, bw = cv2.threshold(fil_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                masks.append(bw)
                self.plot_image(bw)
        return masks

    def find_polynomial(self, thres_coef):
        """
        Находит полиномиальные кривые на масках документа.

        Аргументы:
        - thres_coef: Коэффициент порога для аппроксимации полиномиальной кривой.

        Возвращает:
        - results: Список с результатами, содержащими аппроксимированные полиномиальные кривые, выпуклые оболочки
                 и контуры для каждой маски.
        """

        masks = self.find_document_masks_yolo()
        results = []
        for mask in masks:
            # Находит контуры в маске
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Находит наибольший контур по площади
            big_contour = max(contours, key=cv2.contourArea)
            result = np.zeros_like(self.initial_image)
            cv2.drawContours(result, [big_contour], -1, (0, 255, 0), cv2.FILLED)
            epsilon = thres_coef * cv2.arcLength(big_contour, True)
            # Аппроксимирует контур полиномиальной кривой
            approx = cv2.approxPolyDP(big_contour, epsilon, True)
            # Вычисляет выпуклую оболочку контура
            hull = cv2.convexHull(approx)
            results.append((approx, hull, big_contour))
        return results

    def plot_polynomial(self, concave, conv, cont):
        """
        Визуализирует полиномиальные кривые на изображении документа.

        Аргументы:
        - concave: Список контуров аппроксимированных полиномиальных кривых.
        - conv: Список контуров выпуклых оболочек полиномиальных кривых.
        - cont: Список контуров исходных контуров.

        Вывод:
        - Отображает изображение документа с нарисованными полиномиальными кривыми.
        """

        img = self.initial_image.copy()
        cv2.drawContours(img, conv, -1, (255, 0, 0), 8)  # Выпуклые оболочки - синий цвет
        cv2.drawContours(img, cont, -1, (0, 255, 0), 8)  # Исходные контуры - зеленый цвет
        cv2.drawContours(img, concave, -1, (0, 0, 255), 8)  # Аппроксимированные полиномиальные кривые - красный цвет

        self.plot_image(img)

    @staticmethod
    def sort_points(points):
        """
        Сортирует точки в порядке их угла относительно центра фигуры.

        Аргументы:
        - points: Массив точек для сортировки.

        Возвращает:
        - Отсортированный массив точек.
        """

        figure_center = np.mean(points, axis=0)
        points_list = list(points)
        points_list.sort(key=lambda point: np.arctan2(point[0] - figure_center[0], point[1] - figure_center[1]))
        return np.array(points_list)

    @staticmethod
    def line_point_cross_product(line, point) -> np.ndarray:
        """
        Вычисляет векторное произведение между линией и точкой.

        Аргументы:
        - line: Массив из двух точек, представляющих линию.
        - point: Точка.

        Возвращает:
        - Векторное произведение в виде массива NumPy.
        """

        v1 = np.subtract(line[1], line[0])
        v2 = np.subtract(line[1], point)
        xp = np.cross(v1, v2)
        return xp

    def find_edges(self, contour, convex):
        """
        Определяет края многоугольника на изображении.

        Аргументы:
        - contour: Контур многоугольника.
        - convex: Выпуклая оболочка многоугольника.

        Возвращает:
        - left: Край, лежащий слева от диагональных линий.
        - right: Край, лежащий  справа от диагональных линий.
        - top: Край, лежащий  сверху от диагональных линий.
        - bottom: Край, лежащий  снизу от диагональных линий.
        """

        img_test = self.initial_image.copy()

        # Извлечение граней и углов из массивов
        edges = np.squeeze(np.array(contour), axis=1)
        corners = np.reshape(np.array(convex), (4, 2))

        # Сортировка углов по порядку
        corners = self.sort_points(corners)

        # Формирование диагональных линий
        diag_lines = np.array([[corners[0], corners[2]], [corners[1], corners[3]]])

        # Отрисовка диагональных линий на изображении
        for num, line in enumerate(diag_lines):
            cv2.line(img_test, line[0], line[1], (0, 255 - 100 * num, 0), 2)

        self.plot_image(img_test)

        # Выделение краёв, лежащих слева от диагональных линий
        left = [edge for edge in np.array(edges).tolist() if
                (self.line_point_cross_product(diag_lines[0], edge) < 0 < self.line_point_cross_product(
                    diag_lines[1], edge))]

        # Выделение краёв, лежащих снизу от диагональных линий
        bottom = [edge for edge in np.array(edges).tolist() if
                  (self.line_point_cross_product(diag_lines[0], edge) < 0 and self.line_point_cross_product(
                      diag_lines[1], edge) < 0)]

        # Выделение краёв, лежащих справа от диагональных линий
        right = [edge for edge in np.array(edges).tolist() if
                 (self.line_point_cross_product(diag_lines[0], edge) > 0 > self.line_point_cross_product(
                     diag_lines[1], edge))]

        # Выделение краёв, лежащих сверху от диагональных линий
        top = [edge for edge in np.array(edges).tolist() if
               (self.line_point_cross_product(diag_lines[0], edge) > 0 and self.line_point_cross_product(
                   diag_lines[1], edge) > 0)]

        top.sort()

        right = right[::-1]  # Инвертирование порядка краёв справа

        return left, right, top, bottom

    @staticmethod
    def get_dimensions(left, right, top, bottom):
        """
        Вычисляет размеры многоугольника на изображении.

        Аргументы:
        - left: Край, лежащий слева от диагональных линий.
        - right: Край, лежащий справа от диагональных линий.
        - top: Край, лежащий сверху от диагональных линий.
        - bottom: Край, лежащий снизу от диагональных линий.

        Возвращает:
        - length: Длина многоугольника.
        - width: Ширина многоугольника.
        """

        # Вычисление длины краёв
        left_length = round(spatial.distance.euclidean(left[0], left[-1]))
        right_length = round(spatial.distance.euclidean(right[0], right[-1]))
        top_length = round(spatial.distance.euclidean(top[0], top[-1]))
        bottom_length = round(spatial.distance.euclidean(bottom[0], bottom[-1]))

        # Вычисление средних значений длины и ширины
        length = int(np.mean([left_length, right_length]))
        width = int(np.mean([top_length, bottom_length]))

        return length, width

    def plot_colorful_edges(self, left, right, top, bottom):
        """
        Визуализирует грани многоугольника на изображении разными цветами.

        Аргументы:
        - left: Грани, лежащие слева от диагональных линий.
        - right: Грани, лежащие справа от диагональных линий.
        - top: Грани, расположенные сверху от диагональных линий.
        - bottom: Грани, расположенные снизу от диагональных линий.
        """

        img = self.initial_image.copy()
        for point in left:
            cv2.circle(img, point, 2, [0, 0, 255], -1)
        for point in bottom:
            cv2.circle(img, point, 2, [0, 255, 0], -1)
        for point in right:
            cv2.circle(img, point, 2, [255, 0, 0], -1)
        for point in top:
            cv2.circle(img, point, 2, [0, 255, 255], -1)
        self.plot_image(img)

    @staticmethod
    def interpolate(x, y, n_points):
        """
        Интерполирует значения координат x и y для генерации новых точек.

        Аргументы:
        - x: Список значений координаты x.
        - y: Список значений координаты y.
        - n_points: Желаемое количество новых точек для интерполяции.

        Возвращает:
        - x1: Интерполированные значения координаты x.
        - y1: Интерполированные значения координаты y.
        """

        interp_x = interp1d(np.arange(len(x)), x)
        interp_y = interp1d(np.arange(len(y)), y)
        new_indices = np.linspace(0, len(x) - 1, n_points)
        x1 = interp_x(new_indices)
        y1 = interp_y(new_indices)
        return x1, y1

    def calculate_opposite_interpolate(self, side_1, side_2, smooth=False):
        """
        Вычисляет интерполированные координаты противоположных сторон.

        Аргументы:
        - side_1: Список координат первой стороны.
        - side_2: Список координат второй стороны.
        - smooth: Флаг, указывающий на необходимость сглаживания данных.

        Возвращает:
        - new_side_1_x: Интерполированные значения координаты x для первой стороны.
        - new_side_1_y: Интерполированные значения координаты y для первой стороны.
        - new_side_2_x: Интерполированные значения координаты x для второй стороны.
        - new_side_2_y: Интерполированные значения координаты y для второй стороны.
        """

        side_1_x, side_1_y = zip(*side_1)
        side_2_x, side_2_y = zip(*side_2)
        pts_num_lr = np.max((len(side_1_x), len(side_2_x)))

        new_side_1_x, new_side_1_y = self.interpolate(side_1_x, side_1_y, pts_num_lr)
        new_side_2_x, new_side_2_y = self.interpolate(side_2_x, side_2_y, pts_num_lr)

        if smooth:
            window = 51
            coef = 5
            new_side_1_x = savgol_filter(new_side_1_x, window, coef)
            new_side_1_y = savgol_filter(new_side_1_y, window, coef)
            new_side_2_x = savgol_filter(new_side_2_x, window, coef)
            new_side_2_y = savgol_filter(new_side_2_y, window, coef)
        return new_side_1_x, new_side_1_y, new_side_2_x, new_side_2_y

    def get_lines_of_approximation(self, num, l_x, l_y, r_x, r_y, b_x, b_y, t_x, t_y):
        """
        Получает линии аппроксимации для указанного числа точек.

        Аргументы:
        - num: Число точек для аппроксимации.
        - l_x: Интерполированные значения координаты x для левой стороны.
        - l_y: Интерполированные значения координаты y для левой стороны.
        - r_x: Интерполированные значения координаты x для правой стороны.
        - r_y: Интерполированные значения координаты y для правой стороны.
        - b_x: Интерполированные значения координаты x для нижней стороны.
        - b_y: Интерполированные значения координаты y для нижней стороны.
        - t_x: Интерполированные значения координаты x для верхней стороны.
        - t_y: Интерполированные значения координаты y для верхней стороны.

        Возвращает:
        - linfit_x_bt: Функция аппроксимации для координаты x по горизонтали.
        - linfit_y_bt: Функция аппроксимации для координаты y по горизонтали.
        - linfit_x_lr: Функция аппроксимации для координаты x по вертикали.
        - linfit_y_lr: Функция аппроксимации для координаты y по вертикали.
        """

        linfit_x_lr = interp1d([1, num], np.vstack((l_x, r_x)), axis=0)
        linfit_y_lr = interp1d([1, num], np.vstack((l_y, r_y)), axis=0)

        linfit_x_bt = interp1d([1, num], np.vstack((b_x, t_x)), axis=0)
        linfit_y_bt = interp1d([1, num], np.vstack((b_y, t_y)), axis=0)

        for i in range(1, num + 1):
            plt.imshow(self.initial_image)
            plt.plot(linfit_x_bt(i), linfit_y_bt(i))
            plt.plot(linfit_x_lr(i), linfit_y_lr(i))
        plt.show()
        return linfit_x_bt, linfit_y_bt, linfit_x_lr, linfit_y_lr

    @staticmethod
    def calculate_approximation(p, x, y, shape_flag):
        """
        Вычисляет аппроксимацию для заданных коэффициентов и координат.

        Аргументы:
        - p: Коэффициенты аппроксимации.
        - x: Координата x.
        - y: Координата y.
        - shape_flag: Флаг формы (0 для горизонтальной аппроксимации, 1 для вертикальной).

        Возвращает:
        - Разность между аппроксимацией и исходными координатами.
        """

        if shape_flag == 0:
            return p[0] * y ** 3 + p[1] * y ** 2 + p[2] * y + p[3] - x
        else:
            return p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3] - y

    def get_interpolation_approximation(self, shape, num_ln, degree, num_p, linfit_x, linfit_y):
        """
        Вычисляет аппроксимацию интерполяции.

        Аргументы:
        - shape: Размерность изображения (высота или ширина, в зависимости от флага формы).
        - num_ln: Количество линий интерполяции.
        - degree: Степень аппроксимации.
        - num_p: Количество точек аппроксимации.
        - linfit_x: Функция интерполяции для координаты x.
        - linfit_y: Функция интерполяции для координаты y.

        Возвращает:
        - Список полигонов с аппроксимированными координатами.
        """

        polys_bt = []
        shape_flag = 0 if shape == self.initial_image.shape[0] else 1

        for i in range(1, num_ln + 1):
            x = linfit_x(i)
            y = linfit_y(i)

            p0 = np.zeros(degree + 1)
            result = least_squares(self.calculate_approximation, p0, args=(x, y, shape_flag))
            if shape_flag == 0:
                y_new = np.linspace(0, shape, num_p)
                x_new = result.x[0] * y_new ** 3 + result.x[1] * y_new ** 2 + result.x[2] * y_new + result.x[3]
            else:
                x_new = np.linspace(0, shape, num_p)
                y_new = result.x[0] * x_new ** 3 + result.x[1] * x_new ** 2 + result.x[2] * x_new + result.x[3]

            poly_coor = np.hstack((np.reshape(x_new, (x_new.shape[0], 1)), np.reshape(y_new, (y_new.shape[0], 1))))
            polys_bt.append(poly_coor)
            plt.plot(x_new, y_new)

        return polys_bt

    @staticmethod
    def flatten(t):
        """
        Выполняет сглаживание вложенных списков.

        Аргументы:
        - t: вложенный список

        Возвращает:
        - Одномерный список, полученный путем сглаживания вложенных списков
        """
        return list(itertools.chain.from_iterable(t))

    def find_intersection(self, curve_1, curve_2, tolerance):
        """
        Находит точку пересечения двух кривых.

        Аргументы:
        - curve_1: список координат точек первой кривой
        - curve_2: список координат точек второй кривой
        - tolerance: пороговое значение для определения близости точек

        Возвращает:
        - Координаты точки пересечения двух кривых
        """

        # Строим деревья для каждой кривой
        tree_big = spatial.cKDTree(curve_1)
        tree_small = spatial.cKDTree(curve_2)
        # Выполняем запрос на поиск ближайших соседей между деревьями с использованием порогового значения tolerance
        inds = tree_small.query_ball_tree(tree_big, r=tolerance)
        # Объединяем индексы найденных соседей в один список
        small_inds = self.flatten([i for i in inds if i])
        # Получаем уникальные индексы
        unique_inds = np.unique(small_inds).tolist()
        # Получаем координаты точек из curve_1, соответствующие уникальным индексам
        unique_coords = [curve_1[i] for i in unique_inds]
        return np.mean(np.asarray(unique_coords), axis=0)

    def get_intersection_interpolation_approximation(self, length, width, polys_bt, polys_lr):
        """
        Вычисляет пересечения между линиями интерполяции аппроксимации.

        Аргументы:
        - length: длина изображения
        - width: ширина изображения
        - polys_bt: список полиномиальных координат для линий, идущих в направлении bottom-top
        - polys_lr: список полиномиальных координат для линий, идущих в направлении left-right

        Возвращает:
        - Список координат пересечений линий интерполяции аппроксимации
        """

        plt.imshow(self.initial_image)
        points_set = []
        # Вычисление значения tolerance как округленного значения 1% от длины или ширины изображения
        # (в зависимости от того, какая величина больше)
        tolerance = ceil(length * 0.01) if length > width else ceil(width * 0.01)
        if tolerance < 20:
            tolerance = 20.0
        print(f'TOLERANCE {tolerance}')
        for line1, line2 in itertools.product(polys_bt, polys_lr):
            # Поиск пересечения между текущими линиями line1 и line2, используя заданный tolerance
            a = self.find_intersection(line1, line2, tolerance)
            points_set.append([a[0], a[1]])
            plt.plot(round(a[0]), round(a[1]), 'ro')
        plt.show()
        return points_set

    def get_fix_grid_coordinates(self, length, width, num_of_line):
        """
        Вычисляет координаты фиксированной сетки на изображении.

        Аргументы:
        - length: длина изображения
        - width: ширина изображения
        - num_of_line: количество линий в сетке

        Возвращает:
        - Массив координат фиксированной сетки

        """

        x = np.linspace(0, width, num_of_line)
        y = np.linspace(0, length, num_of_line)
        xv, yv = np.meshgrid(x, y)
        XYpairs = np.dstack([xv, yv]).reshape(-1, 2)
        plt.imshow(self.initial_image)
        plt.plot(XYpairs[:, 0], XYpairs[:, 1], marker='.', color='k', linestyle='none')
        plt.show()
        return XYpairs

    @staticmethod
    def crop_image(img):
        """
        Обрезает изображение в соответствии с контурами на нем.

        Параметры:
        - img: исходное изображение (массив пикселей).

        Возвращает:
        - cropped_img: обрезанное изображение.
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Применение пороговой обработки для получения бинарного изображения
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # Поиск контуров на бинарном изображении
        contour = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0] if len(contour) == 2 else contour[1]
        # Сортировка контуров по площади в порядке убывания
        contour = sorted(contour, key=cv2.contourArea, reverse=True)
        if len(contour) > 0:
            # Получение ограничивающего прямоугольника для самого большого контура
            x, y, w, h = cv2.boundingRect(contour[0])
            # Обрезка изображения согласно ограничивающему прямоугольнику
            img = img[y:y + h, x:x + w]
        return img

    def remap_image(self, height, width, points_set, xy_pairs):
        """
        Переотображает исходное изображение с использованием заданных точек.

        Параметры:
        - height: высота переотображаемого изображения.
        - width: ширина переотображаемого изображения.
        - points_set: список точек, определяющих переотображение.
        - xy_pairs: координаты сетки для переотображения.

        Возвращает:
        - crop_warped_image_BGR: переотображенное и обрезанное изображение в формате BGR.
        """

        img = self.initial_image.copy()
        # Создание комплексных чисел для формирования сетки
        comp_w = complex(f'{width}j')
        comp_h = complex(f'{height}j')
        # Создание сетки
        grid_x, grid_y = np.mgrid[0:(height - 1):comp_h, 0:(width - 1):comp_w]
        # Переотображение точек на сетку с использованием кубической интерполяции
        grid_z = griddata(np.fliplr(xy_pairs), np.fliplr(points_set), (grid_x, grid_y), method='cubic')
        # Извлечение координат x и y из переотображенных точек
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width)
        # Преобразование координат в тип float32
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        # Переотображение исходного изображения с использованием полученных координат
        warped_image = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
        warped_image = np.fliplr(np.rot90(warped_image, 2))
        # Обрезка переотображенного изображения
        # crop_warped_image = self.crop_image(warped_image)
        crop_warped_image_BGR = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
        plt.subplot(121, title='before')
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(122, title='after')
        plt.imshow(warped_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return crop_warped_image_BGR
