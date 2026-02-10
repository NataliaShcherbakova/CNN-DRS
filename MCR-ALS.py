"""
Обработка спектральных данных MCR-ALS анализом

Для экспериментальных данных в основном каталоге получены следующие значения LoF (lack of fit):
LoF MCR-0 для Air: 0.310%
LoF MCR-0 для Sp10: 0.241%
LoF MCR-0 для Sp40: 0.365%
LoF MCR-0 для Sp80: 0.540%
LoF MCR-0 для Sp100: 0.697%
LoF MCR-0 для Trans: 2.956%

Программа загружает спектры, выполняет MCR-ALS разложение с фиксированными компонентами,
и вычисляет спектры чистых компонентов.
"""

import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.linalg import svd
import warnings

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')


# ============================================================================
# Функции загрузки и предобработки данных
# ============================================================================

def load_spectra(filename):
    """
    Загружает спектральные данные из Excel-файла.

    Ожидаемая структура файла:
    - Первый столбец: 'Depth' (глубина)
    - Остальные столбцы: спектры по длинам волн

    Возвращает:
    depths - массив глубин
    wav - массив длин волн
    X - матрица спектров (образцы × длины волн)
    """
    try:
        # Читаем данные
        df = pd.read_excel(filename)

        # Конвертируем все числовые колонки (на случай если числа читаются как текст)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Извлекаем глубины и длины волн
        depths = df['Depth'].values
        wav = df.columns[2:].values.astype(float)  # предполагаем, что длины волн с 3-й колонки

        # Матрица спектров
        X = df.iloc[:, 2:].values

        # Удаляем строки с пропущенными глубинами
        valid_mask = ~np.isnan(depths)
        depths = depths[valid_mask]
        X = X[valid_mask]

        # Удаляем строки где все спектры NaN
        spectral_mask = ~np.all(np.isnan(X), axis=1)
        depths = depths[spectral_mask]
        X = X[spectral_mask]

        # Заменяем оставшиеся NaN на нули
        X = np.nan_to_num(X)

        return depths, wav, X

    except Exception as e:
        print(f"Ошибка при загрузке файла {filename}: {e}")
        return np.array([]), np.array([]), np.array([])


def offset_correction(X):
    """Коррекция базовой линии - вычитаем минимум по каждому спектру."""
    return X - np.min(X, axis=1)[:, None]


# ============================================================================
# MCR-ALS алгоритм с фиксированными компонентами
# ============================================================================

def mcr_als_fixed(X, S_fixed_dict, n_components, init='svd',
                  nonneg_C=True, nonneg_S=True, max_iter=2000, tol=1e-8):
    """
    MCR-ALS разложение с возможностью фиксации некоторых спектров.

    Параметры:
    X - матрица данных (образцы × длины волн)
    S_fixed_dict - словарь {индекс: спектр} для фиксированных компонентов
    n_components - общее число компонентов

    Возвращает:
    C - матрица концентраций (образцы × компоненты)
    S - матрица спектров (компоненты × длины волн)
    """
    # Проверка на пустые данные
    if len(X) == 0:
        return np.zeros((X.shape[0], n_components)), np.zeros((n_components, X.shape[1]))

    n_samples, n_wav = X.shape
    fixed_indices = sorted(S_fixed_dict.keys())
    n_fixed = len(fixed_indices)
    n_free = n_components - n_fixed

    if n_free < 0:
        raise ValueError("Количество фиксированных компонентов превышает общее число компонентов.")

    # Собираем фиксированные спектры в массив
    S_fixed_arr = np.zeros((n_fixed, n_wav))
    for i, idx in enumerate(fixed_indices):
        S_fixed_arr[i] = S_fixed_dict[idx]

    # Индексы свободных компонентов
    free_indices = [i for i in range(n_components) if i not in fixed_indices]

    # Инициализация свободных спектров
    if isinstance(init, str) and init == 'svd':
        # Используем SVD для начального приближения
        U, s, Vt = svd(X, full_matrices=False)
        S_free = np.abs(Vt[:n_free]) if n_free > 0 else np.array([]).reshape(0, n_wav)
    else:
        # Используем предоставленную инициализацию
        S_free = init[free_indices] if init.shape[0] == n_components else init[:n_free]

    # Основной цикл ALS
    for iteration in range(max_iter):
        S_old_free = S_free.copy()

        # Формируем полную матрицу спектров
        S_full = np.zeros((n_components, n_wav))
        for i, idx in enumerate(fixed_indices):
            S_full[idx] = S_fixed_arr[i]
        for i, idx in enumerate(free_indices):
            S_full[idx] = S_free[i]

        # Шаг 1: Оцениваем концентрации (C = X * S^+)
        C = X @ pinv(S_full)
        if nonneg_C:
            C[C < 0] = 0

        # Шаг 2: Оцениваем свободные спектры
        if n_free > 0:
            # Вклад фиксированных компонентов
            C_fixed_part = np.zeros_like(C)
            for i, idx in enumerate(fixed_indices):
                C_fixed_part[:, idx] = C[:, idx]
            X_fixed_contribution = C_fixed_part @ S_full

            # Вычитаем вклад фиксированных компонентов
            X_free = X - X_fixed_contribution
            C_free = C[:, free_indices]

            # Оцениваем свободные спектры
            S_free_new = pinv(C_free) @ X_free
            if nonneg_S:
                S_free_new[S_free_new < 0] = 0

            S_free = S_free_new

        # Проверка сходимости
        if n_free > 0 and norm(S_old_free) > 0:
            diff = norm(S_free - S_old_free) / norm(S_old_free)
            if diff < tol:
                # print(f"Сходимость достигнута на итерации {iteration+1}")
                break

    # Формируем итоговую матрицу спектров
    S_result = np.zeros((n_components, n_wav))
    for i, idx in enumerate(fixed_indices):
        S_result[idx] = S_fixed_arr[i]
    for i, idx in enumerate(free_indices):
        S_result[idx] = S_free[i]

    return C, S_result


def calculate_lof(X, C, S):
    """Вычисление Lack of Fit (LoF) в процентах."""
    if len(X) == 0:
        return np.inf

    X_reconstructed = C @ S
    residual = X - X_reconstructed
    lof = 100 * norm(residual) / norm(X)
    return lof


# ============================================================================
# Основной блок программы
# ============================================================================

if __name__ == "__main__":
    # Список файлов с данными
    files = {
        'Air': 'Air_test.xlsx',
        'Sp10': '10_test.xlsx',
        'Sp40': '40_test.xlsx',
        'Sp80': '80_test.xlsx',
        'Sp100': '100_test.xlsx'
    }
    trans_file = 'TransPE_test.xlsx'

    # Разделение на обучающие и тестовые данные
    training_keys = ['Air', 'Sp10', 'Sp40', 'Sp100']  # для калибровки
    test_key = 'Sp80'  # тестовый набор

    # Загружаем все данные
    print("Загрузка данных...")
    data = {}
    for key, fn in files.items():
        depths, wav, X = load_spectra(fn)
        if len(depths) > 0:
            data[key] = (depths, wav, X)
            print(f"  {key}: {len(depths)} образцов, {len(wav)} длин волн")

    # Загружаем трансмиссионные данные
    trans_depths, wav_trans, X_trans = load_spectra(trans_file)
    if len(trans_depths) > 0:
        X_trans = offset_correction(X_trans)
        print(f"  Trans: {len(trans_depths)} образцов")

    # Определяем общий набор длин волн
    wav = data['Air'][1] if 'Air' in data else wav_trans
    n_wav = len(wav)
    print(f"\nОбщее число длин волн: {n_wav}")

    # ========================================================================
    # Этап 1: Определение спектра поглощения (Absorption)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Этап 1: Калибровка спектра поглощения (Absorption)")
    print("=" * 60)

    sa_list_mcr2 = []  # здесь будем накапливать спектры поглощения

    for key in training_keys:
        if key not in data:
            continue

        depths, _, X = data[key]
        if len(depths) < 3:
            print(f"  Пропускаем {key}: недостаточно образцов")
            continue

        print(f"  Обрабатываю {key}...")

        # 1. Предварительное MCR с 3 компонентами
        C3, S3 = mcr_als_fixed(X, {}, 3, init='svd', nonneg_C=True, nonneg_S=True)

        # 2. Ищем компонент с наименьшей дисперсией (вероятно, transmission)
        stds = np.std(S3, axis=1)
        s1_idx = np.argmin(stds)  # transmission component

        # 3. Фиксируем transmission как единичный спектр
        s1_fixed = np.ones(n_wav)
        S_fixed_mcr2 = {s1_idx: s1_fixed}

        # 4. Повторяем MCR с фиксированным transmission
        C_mcr2, S_mcr2 = mcr_als_fixed(X, S_fixed_mcr2, 3, init=S3,
                                       nonneg_C=True, nonneg_S=True)

        # 5. Нормировка по толстым образцам (depth > 2.5)
        thick_mask = depths > 2.5
        other_indices_mcr2 = [i for i in range(3) if i != s1_idx]

        if np.sum(thick_mask) > 0 and len(other_indices_mcr2) >= 2:
            c2_idx, c3_idx = other_indices_mcr2[0], other_indices_mcr2[1]

            mean_c2_thick = np.mean(C_mcr2[thick_mask, c2_idx])
            mean_c3_thick = np.mean(C_mcr2[thick_mask, c3_idx])

            if mean_c2_thick > 0:
                C_mcr2[:, c2_idx] /= mean_c2_thick
                S_mcr2[c2_idx, :] *= mean_c2_thick

            if mean_c3_thick > 0:
                C_mcr2[:, c3_idx] /= mean_c3_thick
                S_mcr2[c3_idx, :] *= mean_c3_thick

        # 6. Сохраняем спектр поглощения
        sa_idx_mcr2 = other_indices_mcr2[0]
        sa_list_mcr2.append(S_mcr2[sa_idx_mcr2])

    # Усредняем спектры поглощения по всем обучающим наборам
    sa_final = np.mean(sa_list_mcr2, axis=0)
    print(f"  Получен усредненный спектр поглощения")

    # ========================================================================
    # Этап 2: Определение спектра рассеяния (Scattering)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Этап 2: Калибровка спектра рассеяния (Scattering)")
    print("=" * 60)

    ss_list_mcr1 = []  # здесь будем накапливать спектры рассеяния

    for key in training_keys:
        if key not in data:
            continue

        depths, _, X = data[key]
        if len(depths) < 3:
            continue

        print(f"  Обрабатываю {key}...")

        # 1. Предварительное MCR
        C3_temp, S3_temp = mcr_als_fixed(X, {}, 3, init='svd', nonneg_C=True, nonneg_S=True)

        # 2. Определяем индексы компонентов
        stds_temp = np.std(S3_temp, axis=1)
        s1_idx_temp = np.argmin(stds_temp)
        other_indices_mcr2_temp = [i for i in range(3) if i != s1_idx_temp]
        sa_idx_mcr2_temp = other_indices_mcr2_temp[0]
        ss_idx_mcr2_temp = other_indices_mcr2_temp[1]

        # 3. Фиксируем transmission и absorption
        S_fixed_mcr1 = {
            s1_idx_temp: np.ones(n_wav),
            sa_idx_mcr2_temp: sa_final
        }

        # 4. MCR с двумя фиксированными компонентами
        C_mcr1, S_mcr1 = mcr_als_fixed(X, S_fixed_mcr1, 3, init=S3_temp,
                                       nonneg_C=True, nonneg_S=True)

        # 5. Нормировка scattering по толстым образцам
        thick_mask_mcr1 = depths > 2.5
        if np.sum(thick_mask_mcr1) > 0:
            mean_c3_thick_mcr1 = np.mean(C_mcr1[thick_mask_mcr1, ss_idx_mcr2_temp])
            if mean_c3_thick_mcr1 > 0:
                C_mcr1[:, ss_idx_mcr2_temp] /= mean_c3_thick_mcr1
                S_mcr1[ss_idx_mcr2_temp, :] *= mean_c3_thick_mcr1

        # 6. Сохраняем спектр рассеяния
        ss_list_mcr1.append(S_mcr1[ss_idx_mcr2_temp])

    # Усредняем спектры рассеяния
    ss_final = np.mean(ss_list_mcr1, axis=0)
    print(f"  Получен усредненный спектр рассеяния")

    # ========================================================================
    # Этап 3: Применение калиброванной модели ко всем данным
    # ========================================================================
    print("\n" + "=" * 60)
    print("Этап 3: Применение модели MCR-0 ко всем наборам данных")
    print("=" * 60)

    # Финальная матрица спектров
    s1_final = np.ones(n_wav)  # transmission
    S_matrix = np.vstack([s1_final, sa_final, ss_final])

    # Применяем модель к каждому набору данных
    results = {}

    for key in files:
        if key not in data:
            continue

        depths, _, X = data[key]
        if len(depths) < 2:
            continue

        # Проекция на фиксированные спектры
        C_temp = X @ pinv(S_matrix)
        C_final = np.maximum(C_temp, 0)  # неотрицательность

        # Вычисляем ошибку реконструкции
        lof_mcr0 = calculate_lof(X, C_final, S_matrix)
        results[key] = {'depths': depths, 'C': C_final, 'lof': lof_mcr0}

        print(f"  {key}: LoF = {lof_mcr0:.3f}%")

    # ========================================================================
    # Интерполяция на общую сетку глубин и сохранение результатов
    # ========================================================================
    print("\n" + "=" * 60)
    print("Сохранение результатов...")
    print("=" * 60)

    # Создаем общую сетку глубин
    all_depths = np.sort(np.unique(np.concatenate([results[k]['depths'] for k in results])))

    # Готовим DataFrames для сохранения
    df_abs = pd.DataFrame({'Depth': all_depths})
    df_scatt = pd.DataFrame({'Depth': all_depths})
    df_trans = pd.DataFrame({'Depth': all_depths})

    # Интерполяция концентраций на общую сетку
    for key in results:
        depths = results[key]['depths']
        C = results[key]['C']

        # Предполагаем, что C[:,0] - transmission, C[:,1] - absorption, C[:,2] - scattering
        df_abs[key] = np.interp(all_depths, depths, C[:, 1])
        df_scatt[key] = np.interp(all_depths, depths, C[:, 2])
        df_trans[key] = np.interp(all_depths, depths, C[:, 0])

    # Сохраняем концентрации
    df_abs.to_excel('computed_Absorb_test.xlsx', index=False)
    df_scatt.to_excel('computed_Scatt_test.xlsx', index=False)
    df_trans.to_excel('computed_Trans_test.xlsx', index=False)

    # Сохраняем чистые спектры
    df_spec = pd.DataFrame({
        'Wavelength': wav,
        'Transmission': s1_final,
        'Absorption': sa_final,
        'Scattering': ss_final
    })
    df_spec.to_excel('pure_spectra_test.xlsx', index=False)

    # ========================================================================
    # Обработка трансмиссионных данных
    # ========================================================================
    if len(trans_depths) > 0:
        print("\nОбработка трансмиссионных данных...")

        # Проекция на те же спектры
        C_trans = X_trans @ pinv(S_matrix)
        C_trans = np.maximum(C_trans, 0)

        # Вычисление LoF
        lof_trans = calculate_lof(X_trans, C_trans, S_matrix)

    # ========================================================================
    # Финальный вывод результатов
    # ========================================================================
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print("=" * 60)
    print("Значения Lack of Fit (LoF) для MCR-0 модели:")

    for key in files:
        if key in results:
            print(f"  {key}: {results[key]['lof']:.3f}%")

    if len(trans_depths) > 0:
        print(f"  Trans: {lof_trans:.3f}%")

    print("\n" + "=" * 60)
    print("Сохраненные файлы:")
    print("  1. computed_Absorb_test.xlsx - концентрации поглощения")
    print("  2. computed_Scatt_test.xlsx - концентрации рассеяния")
    print("  3. computed_Trans_test.xlsx - концентрации пропускания")
    print("  4. pure_spectra_test.xlsx - чистые спектры компонентов")
    print("=" * 60)
    print("\nОбработка завершена успешно!")