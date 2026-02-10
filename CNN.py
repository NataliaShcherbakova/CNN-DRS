"""
Гибридная модель CNN - оптимальное сочетание скорости и точности
Объединяет лучшие компоненты CNN_Data_synth.py и CNN_R09.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import warnings
import json
from datetime import datetime
from scipy import stats
import gc

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


# ============================================================================
# 1. Функции загрузки данных
# ============================================================================

def load_spectra(file_path):
    """
    Загрузка спектров из Excel файла

    Параметры:
    file_path - путь к файлу

    Возвращает:
    DataFrame с колонками 'h' (толщина) и спектры
    """
    df = pd.read_excel(file_path)

    # Поиск колонки с толщиной
    depth_col = 'Depth' if 'Depth' in df.columns else next(
        (col for col in df.columns if col.lower() in ['d', 'thickness']), None)

    if depth_col is None:
        raise ValueError(f"Не найдена колонка с толщиной в {file_path}")

    # Определение спектральных колонок
    spec_cols = [col for col in df.columns if col not in [depth_col, 'Name', 'Sheet1'] and
                 (isinstance(col, (int, float)) or str(col).replace('.', '', 1).isdigit())]

    # Преобразование числовых данных
    df[spec_cols] = df[spec_cols].apply(pd.to_numeric, errors='coerce')
    df = df[[depth_col] + spec_cols].dropna()
    df.rename(columns={depth_col: 'h'}, inplace=True)

    return df


def load_all_data():
    """
    Загрузка всех данных: экспериментальных и синтетических
    """
    configs = ['Air', 'Sp10', 'Sp40', 'Sp80', 'Sp100', 'TransPE']
    exp_files = ['Air_test.xlsx', '10_test.xlsx', '40_test.xlsx', '80_test.xlsx', '100_test.xlsx', 'TransPE_test.xlsx']
    synth_files = [f'synthetic_data/{c}_synthetic.xlsx' for c in configs]

    exp_dfs = {configs[i]: load_spectra(exp_files[i]) for i in range(len(configs))}
    synth_dfs = {configs[i]: load_spectra(synth_files[i]) for i in range(len(configs))}

    return exp_dfs, synth_dfs


def load_coeffs_and_pure():
    """
    Загрузка коэффициентов и чистых спектров
    """
    absorb_df = pd.read_excel('computed_Absorb_test.xlsx')
    scatt_df = pd.read_excel('computed_Scatt_test.xlsx')
    trans_df = pd.read_excel('computed_Trans_test.xlsx')
    pure_df = pd.read_excel('pure_spectra_test.xlsx')

    wavelengths = pure_df['Wavelength'].values
    s_a = pure_df['Absorption'].values
    s_s = pure_df['Scattering'].values

    coeffs = {}
    available_configs = absorb_df.columns[1:]

    for config in available_configs:
        df_temp = pd.DataFrame({
            'h': absorb_df['Depth'],
            'c_a': absorb_df[config],
            'c_s': scatt_df[config],
            'c_1': trans_df[config]
        }).dropna()
        coeffs[config] = df_temp

    # Создание усредненных данных для TransPE если отсутствует
    if 'TransPE' not in coeffs:
        mean_c_a = absorb_df[available_configs].mean(axis=1)
        mean_c_s = scatt_df[available_configs].mean(axis=1)
        mean_c_1 = trans_df[available_configs].mean(axis=1)

        coeffs['TransPE'] = pd.DataFrame({
            'h': absorb_df['Depth'],
            'c_a': mean_c_a,
            'c_s': mean_c_s,
            'c_1': mean_c_1
        })

    return wavelengths, s_a, s_s, coeffs


def label_data(df, config, s_a, s_s, coeffs):
    """
    Создание меток для данных на основе коэффициентов
    """
    if config not in coeffs:
        raise KeyError(f"Конфигурация {config} не найдена в coeffs")

    coeff_df = coeffs[config]
    h_values = df['h'].values

    # Интерполяция коэффициентов
    c_a_interp = np.interp(h_values, coeff_df['h'].values, coeff_df['c_a'].values, left=0, right=0)
    c_s_interp = np.interp(h_values, coeff_df['h'].values, coeff_df['c_s'].values, left=0, right=0)

    N = len(df)
    mu_a = np.zeros((N, len(s_a)))
    mu_s = np.zeros((N, len(s_s)))

    # Расчет коэффициентов ослабления
    for i in range(N):
        h = h_values[i]
        if h > 0:
            mu_a[i] = (c_a_interp[i] * s_a) / (2 * h)
            mu_s[i] = (c_s_interp[i] * s_s) / h

    mu_a = np.clip(mu_a, 0, np.inf)
    mu_s = np.clip(mu_s, 0, np.inf)

    # Извлечение спектров
    spec_cols = [col for col in df.columns if col != 'h']
    spectra = df[spec_cols].values

    return spectra, mu_a, mu_s


# ============================================================================
# 2. Улучшенный предобработчик (гибридный)
# ============================================================================

class HybridPreprocessor:
    """
    Гибридный предобработчик: адаптивная трансформация + скорость
    """

    def __init__(self, use_fast_transform=True):
        self.scaler_spectra = StandardScaler()
        self.scaler_mu_a = StandardScaler()
        self.scaler_mu_s = StandardScaler()
        self.mu_s_stats = {}
        self.use_fast_transform = use_fast_transform

    def analyze_mu_s(self, mu_s):
        """
        Быстрый анализ распределения μ_s
        """
        mu_s_flat = mu_s.flatten()
        mu_s_flat = mu_s_flat[mu_s_flat > 0]

        stats_dict = {
            'min': mu_s_flat.min(),
            'max': mu_s_flat.max(),
            'mean': mu_s_flat.mean(),
            'std': mu_s_flat.std(),
            'skewness': stats.skew(mu_s_flat),
            'kurtosis': stats.kurtosis(mu_s_flat),
        }
        return stats_dict

    def fast_transform_mu_s(self, mu_s):
        """
        Быстрая комбинированная трансформация
        """
        mu_s_positive = np.clip(mu_s, 0, None)
        return np.log1p(np.sqrt(mu_s_positive + 1e-8))

    def fast_inverse_transform_mu_s(self, mu_s_transformed):
        """
        Быстрое обратное преобразование
        """
        mu_s = np.expm1(mu_s_transformed)
        mu_s = np.square(mu_s) - 1e-8
        return np.clip(mu_s, 0, None)

    def fit(self, spectra, mu_a, mu_s):
        """
        Обучение трансформаций
        """
        print("  Гибридная предобработка...")

        # Быстрый анализ μ_s
        self.mu_s_stats = self.analyze_mu_s(mu_s)
        print(f"  Статистика μ_s: mean={self.mu_s_stats['mean']:.6f}, "
              f"skew={self.mu_s_stats['skewness']:.3f}")

        # Трансформация спектров
        self.scaler_spectra.fit(spectra)
        self.scaler_mu_a.fit(mu_a)

        # Трансформация μ_s
        if self.use_fast_transform:
            mu_s_transformed = self.fast_transform_mu_s(mu_s)
        else:
            # Более сложная трансформация для сильного скоса
            if self.mu_s_stats['skewness'] > 2.0:
                mu_s_transformed = np.log1p(np.sqrt(mu_s + 1e-8))
            else:
                mu_s_transformed = np.log1p(mu_s)

        self.scaler_mu_s.fit(mu_s_transformed)

    def transform(self, spectra, mu_a, mu_s):
        """
        Применение трансформаций
        """
        spectra_proc = self.scaler_spectra.transform(spectra)
        mu_a_proc = self.scaler_mu_a.transform(mu_a)

        if self.use_fast_transform:
            mu_s_transformed = self.fast_transform_mu_s(mu_s)
        else:
            mu_s_transformed = np.log1p(mu_s)

        mu_s_proc = self.scaler_mu_s.transform(mu_s_transformed)

        return spectra_proc, mu_a_proc, mu_s_proc

    def inverse_transform(self, spectra_norm=None, mu_a_norm=None, mu_s_norm=None):
        """
        Обратное преобразование
        """
        result = {}

        if spectra_norm is not None:
            result['spectra'] = self.scaler_spectra.inverse_transform(spectra_norm)

        if mu_a_norm is not None:
            result['mu_a'] = self.scaler_mu_a.inverse_transform(mu_a_norm)

        if mu_s_norm is not None:
            mu_s_scaled = self.scaler_mu_s.inverse_transform(mu_s_norm)
            if self.use_fast_transform:
                result['mu_s'] = self.fast_inverse_transform_mu_s(mu_s_scaled)
            else:
                result['mu_s'] = np.expm1(mu_s_scaled)

        return result


# ============================================================================
# 3. Гибридная архитектура модели
# ============================================================================

class HybridResidualBlock(nn.Module):
    """
    Оптимизированный остаточный блок
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1, use_depthwise=False):
        super().__init__()

        # Depthwise separable свертки для ускорения
        if use_depthwise:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 3, stride=stride, padding=1,
                          groups=in_channels, bias=False),
                nn.Conv1d(in_channels, out_channels, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride,
                                   padding=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class EfficientSpectralAttention(nn.Module):
    """
    Эффективный механизм внимания
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        attention = avg_out.view(b, c, 1)

        return x * attention.expand_as(x)


class HybridDRSCNN(nn.Module):
    """
    Гибридная модель: скорость CNN_R09 + точность CNN_Data_synth
    """

    def __init__(self, input_channels=591, output_size=591, use_depthwise=False):
        super().__init__()

        # Начальные слои (быстрые)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(2)

        # Оптимизированные остаточные блоки
        self.resblock1 = HybridResidualBlock(32, 64, stride=2, use_depthwise=use_depthwise)
        self.resblock2 = HybridResidualBlock(64, 128, stride=2, use_depthwise=use_depthwise)

        # Эффективный механизм внимания
        self.attention = EfficientSpectralAttention(128)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Головы для предсказания μ_a и μ_s
        self.fc_mu_a = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )

        self.fc_mu_s = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Инициализация весов сети
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Общий путь
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)

        # Внимание
        x = self.attention(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Раздельные пути для μ_a и μ_s
        mu_a = self.fc_mu_a(x)
        mu_s = self.fc_mu_s(x)

        return mu_a, mu_s


# ============================================================================
# 4. Улучшенный датасет
# ============================================================================

class HybridDRSDataset(Dataset):
    """
    Гибридный датасет с умной аугментацией
    """

    def __init__(self, spectra, mu_a, mu_s, augment_prob=0.5, use_smart_aug=True):
        self.spectra = spectra
        self.mu_a = mu_a
        self.mu_s = mu_s
        self.augment_prob = augment_prob
        self.use_smart_aug = use_smart_aug

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx].copy()
        mu_a = self.mu_a[idx].copy()
        mu_s = self.mu_s[idx].copy()

        # Умная аугментация
        if self.use_smart_aug and np.random.random() < self.augment_prob:
            # Гауссовский шум
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, spectrum.shape)
            spectrum = spectrum + noise

            # Случайное масштабирование
            scale = np.random.uniform(0.98, 1.02)
            spectrum = spectrum * scale

            # Случайный сдвиг
            if np.random.random() < 0.1:
                shift = np.random.uniform(-0.01, 0.01)
                spectrum = spectrum + shift

        # Конвертация в тензоры
        spectrum_tensor = torch.FloatTensor(spectrum).unsqueeze(0)
        mu_a_tensor = torch.FloatTensor(mu_a)
        mu_s_tensor = torch.FloatTensor(mu_s)

        return spectrum_tensor, mu_a_tensor, mu_s_tensor


# ============================================================================
# 5. Гибридная функция потерь
# ============================================================================

class HybridLoss(nn.Module):
    """
    Гибридная функция потерь: адаптивная + быстрая
    """

    def __init__(self, adaptive_weights=True, huber_beta=1.0):
        super().__init__()
        self.adaptive_weights = adaptive_weights

        if adaptive_weights:
            self.weights = nn.Parameter(torch.tensor([0.3, 0.7], dtype=torch.float32))
        else:
            self.weights = torch.tensor([0.3, 0.7], dtype=torch.float32)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=huber_beta) if hasattr(nn, 'HuberLoss') else self._huber_loss

    def _huber_loss(self, pred, target, delta=1.0):
        """
        Реализация Huber loss если нет встроенной
        """
        error = torch.abs(pred - target)
        quadratic = torch.clamp(error, max=delta)
        linear = error - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return loss.mean()

    def forward(self, pred_a, pred_s, target_a, target_s):
        # MSE для μ_a
        loss_a = self.mse_loss(pred_a, target_a)

        # Комбинированная потеря для μ_s
        loss_s_huber = self.huber_loss(pred_s, target_s)
        loss_s_mae = self.mae_loss(pred_s, target_s)
        loss_s = 0.8 * loss_s_huber + 0.2 * loss_s_mae

        # Взвешивание потерь
        if self.adaptive_weights:
            total_loss = self.weights[0] * loss_a + self.weights[1] * loss_s
        else:
            total_loss = 0.3 * loss_a + 0.7 * loss_s

        return total_loss, loss_a.item(), loss_s.item()


# ============================================================================
# 6. Гибридное обучение (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================================================

def train_hybrid_model(model, train_loader, val_loader, epochs=180, lr=0.001,
                       use_mixed_precision=True):
    """
    Гибридное обучение: mixed precision + умные техники
    """
    model.to(device)

    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Комбинированный scheduler
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    criterion = HybridLoss(adaptive_weights=True, huber_beta=0.5)

    # Mixed precision scaler
    scaler = GradScaler() if (use_mixed_precision and torch.cuda.is_available()) else None

    train_losses = []
    val_losses = []
    train_losses_a = []
    train_losses_s = []
    val_losses_a = []
    val_losses_s = []

    # ИСПРАВЛЕНИЕ 1: Инициализация best_r2_s как -inf вместо 0
    best_r2_s = -float('inf')
    best_val_loss = float('inf')

    # ИСПРАВЛЕНИЕ 2: Добавляем период "разогрева"
    warmup_epochs = 15  # Первые 15 эпох - разогрев без ранней остановки
    patience_counter = 0
    patience = 30

    print(f"\nГибридное обучение на {epochs} эпох")
    print(f"Mixed precision: {scaler is not None}")
    print(f"Период разогрева: {warmup_epochs} эпох")

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        train_loss_a = 0
        train_loss_s = 0

        for spectra, mu_a, mu_s in train_loader:
            spectra, mu_a, mu_s = spectra.to(device), mu_a.to(device), mu_s.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None and torch.cuda.is_available():
                with autocast():
                    pred_a, pred_s = model(spectra)
                    loss, loss_a, loss_s = criterion(pred_a, pred_s, mu_a, mu_s)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_a, pred_s = model(spectra)
                loss, loss_a, loss_s = criterion(pred_a, pred_s, mu_a, mu_s)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            train_loss_a += loss_a
            train_loss_s += loss_s

        # Обновление learning rate
        cosine_scheduler.step()

        # Валидация
        model.eval()
        val_loss = 0
        val_loss_a = 0
        val_loss_s = 0
        all_pred_s = []
        all_true_s = []

        with torch.no_grad():
            for spectra, mu_a, mu_s in val_loader:
                spectra, mu_a, mu_s = spectra.to(device), mu_a.to(device), mu_s.to(device)
                pred_a, pred_s = model(spectra)

                loss, loss_a, loss_s = criterion(pred_a, pred_s, mu_a, mu_s)

                val_loss += loss.item()
                val_loss_a += loss_a
                val_loss_s += loss_s

                all_pred_s.append(pred_s.cpu().numpy())
                all_true_s.append(mu_s.cpu().numpy())

        # Вычисление средних потерь
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_losses_a.append(train_loss_a / len(train_loader))
        train_losses_s.append(train_loss_s / len(train_loader))
        val_losses_a.append(val_loss_a / len(val_loader))
        val_losses_s.append(val_loss_s / len(val_loader))

        # Вычисление R² на валидации
        if len(all_pred_s) > 0:
            pred_s_flat = np.vstack(all_pred_s).flatten()
            true_s_flat = np.vstack(all_true_s).flatten()
            val_r2_s = r2_score(true_s_flat, pred_s_flat)
        else:
            val_r2_s = 0

        # Обновление scheduler на основе потерь валидации
        plateau_scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Логирование прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_weights = criterion.weights.data.tolist() if criterion.adaptive_weights else [0.3, 0.7]

            print(f"\nЭпоха [{epoch + 1:3d}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} (μ_a: {train_losses_a[-1]:.4f}, μ_s: {train_losses_s[-1]:.4f})")
            print(f"  Val Loss:   {avg_val_loss:.4f} (μ_a: {val_losses_a[-1]:.4f}, μ_s: {val_losses_s[-1]:.4f})")
            print(f"  Val R² μ_s: {val_r2_s:.4f}")
            print(f"  LR: {current_lr:.6f}, Weights: [{current_weights[0]:.2f}, {current_weights[1]:.2f}]")

        # ИСПРАВЛЕНИЕ 3: Правильная логика сохранения и ранней остановки
        # Сохраняем лучшую модель всегда при улучшении R²
        if val_r2_s > best_r2_s:
            best_r2_s = val_r2_s
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_r2_s': val_r2_s,
                'scheduler_cosine': cosine_scheduler.state_dict(),
                'scheduler_plateau': plateau_scheduler.state_dict(),
            }, 'best_hybrid_model.pth')

            # ИСПРАВЛЕНИЕ 4: Сбрасываем счетчик терпения ТОЛЬКО после периода разогрева
            if epoch >= warmup_epochs:
                patience_counter = 0
            print(f"  Сохранена лучшая модель (R² μ_s: {val_r2_s:.4f})")
        else:
            # ИСПРАВЛЕНИЕ 5: Увеличиваем счетчик терпения ТОЛЬКО после периода разогрева
            if epoch >= warmup_epochs:
                patience_counter += 1

        # Ранняя остановка (ТОЛЬКО после периода разогрева)
        if epoch >= warmup_epochs and patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch + 1}")
            print(f"  Лучший R² μ_s: {best_r2_s:.4f}")
            print(f"  Без улучшений в течение {patience_counter} эпох")
            break

    # Загрузка лучшей модели
    if os.path.exists('best_hybrid_model.pth'):
        checkpoint = torch.load('best_hybrid_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nЗагружена лучшая модель (Val R² μ_s: {checkpoint['val_r2_s']:.4f})")

    return train_losses, val_losses, train_losses_a, train_losses_s, val_losses_a, val_losses_s


# ============================================================================
# 7. Оценка модели
# ============================================================================

def evaluate_hybrid_model(model, test_loader, preprocessor):
    """
    Оценка гибридной модели
    """
    model.eval()
    all_pred_a, all_pred_s = [], []
    all_true_a, all_true_s = [], []

    with torch.no_grad():
        for spectra, mu_a, mu_s in test_loader:
            spectra = spectra.to(device)
            pred_a, pred_s = model(spectra)

            all_pred_a.append(pred_a.cpu().numpy())
            all_pred_s.append(pred_s.cpu().numpy())
            all_true_a.append(mu_a.numpy())
            all_true_s.append(mu_s.numpy())

    # Объединение всех батчей
    pred_a = np.vstack(all_pred_a)
    pred_s = np.vstack(all_pred_s)
    true_a = np.vstack(all_true_a)
    true_s = np.vstack(all_true_s)

    # Обратное преобразование к исходному масштабу
    denorm = preprocessor.inverse_transform(mu_a_norm=pred_a, mu_s_norm=pred_s)
    pred_a_denorm = denorm['mu_a']
    pred_s_denorm = denorm['mu_s']

    denorm_true = preprocessor.inverse_transform(mu_a_norm=true_a, mu_s_norm=true_s)
    true_a_denorm = denorm_true['mu_a']
    true_s_denorm = denorm_true['mu_s']

    # Вычисление метрик качества
    r2_a = r2_score(true_a_denorm, pred_a_denorm)
    r2_s = r2_score(true_s_denorm, pred_s_denorm)

    rmse_a = np.sqrt(mean_squared_error(true_a_denorm, pred_a_denorm))
    rmse_s = np.sqrt(mean_squared_error(true_s_denorm, pred_s_denorm))

    mae_a = mean_absolute_error(true_a_denorm, pred_a_denorm)
    mae_s = mean_absolute_error(true_s_denorm, pred_s_denorm)

    # Используем симметричную относительную ошибку (sMAPE) и исключаем выбросы

    # Для μ_s
    valid_idx_s = true_s_denorm > 1e-4  # Более строгий порог
    if np.any(valid_idx_s):
        # Симметричная относительная ошибка (менее чувствительна к малым значениям)
        rel_error_s = np.mean(
            2 * np.abs(pred_s_denorm[valid_idx_s] - true_s_denorm[valid_idx_s]) /
            (np.abs(pred_s_denorm[valid_idx_s]) + np.abs(true_s_denorm[valid_idx_s]) + 1e-8)
        ) * 100

        # Также считаем медианную относительную ошибку (более устойчивую к выбросам)
        median_rel_error_s = np.median(
            2 * np.abs(pred_s_denorm[valid_idx_s] - true_s_denorm[valid_idx_s]) /
            (np.abs(pred_s_denorm[valid_idx_s]) + np.abs(true_s_denorm[valid_idx_s]) + 1e-8)
        ) * 100
    else:
        rel_error_s = 0
        median_rel_error_s = 0

    # Для μ_a
    valid_idx_a = true_a_denorm > 1e-4  # Более строгий порог
    if np.any(valid_idx_a):
        rel_error_a = np.mean(
            2 * np.abs(pred_a_denorm[valid_idx_a] - true_a_denorm[valid_idx_a]) /
            (np.abs(pred_a_denorm[valid_idx_a]) + np.abs(true_a_denorm[valid_idx_a]) + 1e-8)
        ) * 100

        median_rel_error_a = np.median(
            2 * np.abs(pred_a_denorm[valid_idx_a] - true_a_denorm[valid_idx_a]) /
            (np.abs(pred_a_denorm[valid_idx_a]) + np.abs(true_a_denorm[valid_idx_a]) + 1e-8)
        ) * 100
    else:
        rel_error_a = 0
        median_rel_error_a = 0

    print(f"\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ГИБРИДНОЙ МОДЕЛИ")
    print("=" * 60)
    print(f"R² μ_a:   {r2_a:.6f} | R² μ_s:   {r2_s:.6f}")
    print(f"RMSE μ_a: {rmse_a:.6f} | RMSE μ_s: {rmse_s:.6f}")
    print(f"MAE μ_a:  {mae_a:.6f} | MAE μ_s:  {mae_s:.6f}")
    print(f"Средняя симметричная отн. ошибка μ_a: {rel_error_a:.2f}% | μ_s: {rel_error_s:.2f}%")
    print(f"Медианная симметричная отн. ошибка μ_a: {median_rel_error_a:.2f}% | μ_s: {median_rel_error_s:.2f}%")
    print("=" * 60)

    # Статистика предсказаний для μ_s
    print(f"\nСтатистика μ_s:")
    print(f"  Истинные: Min={true_s_denorm.min():.6f}, Max={true_s_denorm.max():.6f}")
    print(f"            Mean={true_s_denorm.mean():.6f}, Std={true_s_denorm.std():.6f}")
    print(f"  Предсказания: Min={pred_s_denorm.min():.6f}, Max={pred_s_denorm.max():.6f}")
    print(f"                Mean={pred_s_denorm.mean():.6f}, Std={pred_s_denorm.std():.6f}")

    # Дополнительная статистика для диагностики
    print(f"\nДиагностика μ_s:")
    print(f"  Количество значений > 1e-4: {np.sum(true_s_denorm > 1e-4)} из {true_s_denorm.size}")
    print(f"  Процент малых значений (<1e-4): {np.sum(true_s_denorm < 1e-4) / true_s_denorm.size * 100:.2f}%")

    # Проверка на выбросы
    q75_s, q25_s = np.percentile(true_s_denorm.flatten(), [75, 25])
    iqr_s = q75_s - q25_s
    upper_bound_s = q75_s + 1.5 * iqr_s
    outliers_s = np.sum(true_s_denorm > upper_bound_s)
    print(f"  Выбросов (> Q3 + 1.5*IQR): {outliers_s}")

    # Статистика предсказаний для μ_a
    print(f"\nСтатистика μ_a:")
    print(f"  Истинные: Min={true_a_denorm.min():.6f}, Max={true_a_denorm.max():.6f}")
    print(f"            Mean={true_a_denorm.mean():.6f}, Std={true_a_denorm.std():.6f}")
    print(f"  Предсказания: Min={pred_a_denorm.min():.6f}, Max={pred_a_denorm.max():.6f}")
    print(f"                Mean={pred_a_denorm.mean():.6f}, Std={pred_a_denorm.std():.6f}")

    return {
        'r2_a': r2_a, 'r2_s': r2_s,
        'rmse_a': rmse_a, 'rmse_s': rmse_s,
        'mae_a': mae_a, 'mae_s': mae_s,
        'rel_error_a': rel_error_a, 'rel_error_s': rel_error_s,
        'median_rel_error_a': median_rel_error_a, 'median_rel_error_s': median_rel_error_s,
        'pred_a': pred_a_denorm, 'pred_s': pred_s_denorm,
        'true_a': true_a_denorm, 'true_s': true_s_denorm
    }

# ============================================================================
# 8. Основная функция
# ============================================================================

def run_hybrid_training(use_depthwise=True, use_mixed_precision=True):
    """
    Запуск гибридного обучения
    """
    print("=" * 60)
    print("ГИБРИДНАЯ МОДЕЛЬ - ОПТИМАЛЬНОЕ РЕШЕНИЕ")
    print("=" * 60)
    print(f"Depthwise convolutions: {use_depthwise}")
    print(f"Mixed precision: {use_mixed_precision}")

    try:
        # 1. Загрузка данных
        print("\n1. Загрузка данных...")
        exp_dfs, synth_dfs = load_all_data()
        wavelengths, s_a, s_s, coeffs = load_coeffs_and_pure()

        print(f"Загружено конфигураций: {len(exp_dfs)}")

        # 2. Подготовка датасета
        print("\n2. Подготовка датасета...")

        def prepare_dataset():
            """
            Подготовка датасета с двойной стратификацией (конфигурация + тип данных)
            """
            all_spectra, all_mu_a, all_mu_s = [], [], []
            all_configs = []
            all_data_types = []  # 'exp' или 'synth'

            configs = list(exp_dfs.keys())
            for config in configs:
                # Экспериментальные данные
                exp_df = exp_dfs[config]
                exp_spectra, exp_mu_a, exp_mu_s = label_data(exp_df, config, s_a, s_s, coeffs)
                n_exp = len(exp_spectra)
                all_spectra.append(exp_spectra)
                all_mu_a.append(exp_mu_a)
                all_mu_s.append(exp_mu_s)
                all_configs.extend([config] * n_exp)
                all_data_types.extend(['exp'] * n_exp)

                # Синтетические данные
                synth_df = synth_dfs[config]
                synth_spectra, synth_mu_a, synth_mu_s = label_data(synth_df, config, s_a, s_s, coeffs)
                n_synth = len(synth_spectra)
                all_spectra.append(synth_spectra)
                all_mu_a.append(synth_mu_a)
                all_mu_s.append(synth_mu_s)
                all_configs.extend([config] * n_synth)
                all_data_types.extend(['synth'] * n_synth)

            # Объединение
            spectra = np.vstack(all_spectra)
            mu_a = np.vstack(all_mu_a)
            mu_s = np.vstack(all_mu_s)
            config_labels = np.array(all_configs)
            data_type_labels = np.array(all_data_types)

            # Создаем комбинированные метки для стратификации
            combined_labels = []
            for config, dtype in zip(config_labels, data_type_labels):
                combined_labels.append(f"{config}_{dtype}")
            combined_labels = np.array(combined_labels)

            print(f"Всего образцов: {len(spectra)}")
            print(f"Уникальных комбинаций (конфигурация_тип): {len(np.unique(combined_labels))}")

            # Стратифицированное разделение по комбинированным меткам
            train_spectra, temp_spectra, train_mu_a, temp_mu_a, train_mu_s, temp_mu_s, train_combined, temp_combined = train_test_split(
                spectra, mu_a, mu_s, combined_labels,
                test_size=0.2,
                random_state=42,
                stratify=combined_labels
            )

            val_spectra, test_spectra, val_mu_a, test_mu_a, val_mu_s, test_mu_s, val_combined, test_combined = train_test_split(
                temp_spectra, temp_mu_a, temp_mu_s, temp_combined,
                test_size=0.5,
                random_state=42,
                stratify=temp_combined
            )

            # Восстанавливаем отдельные метки из комбинированных
            def split_combined_labels(combined_arr):
                configs_split = []
                types_split = []
                for label in combined_arr:
                    config, dtype = label.split('_')
                    configs_split.append(config)
                    types_split.append(dtype)
                return np.array(configs_split), np.array(types_split)

            train_configs, train_types = split_combined_labels(train_combined)
            val_configs, val_types = split_combined_labels(val_combined)
            test_configs, test_types = split_combined_labels(test_combined)

            # Детальный анализ распределения
            print("\n" + "=" * 60)
            print("ДЕТАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ПО КОНФИГУРАЦИЯМ И ТИПАМ ДАННЫХ")
            print("=" * 60)

            for name, data, configs_arr, types_arr in [
                ("Обучающая", train_spectra, train_configs, train_types),
                ("Валидационная", val_spectra, val_configs, val_types),
                ("Тестовая", test_spectra, test_configs, test_types)
            ]:
                print(f"\n{name} выборка ({len(data)} образцов):")

                # По конфигурациям
                unique_configs, config_counts = np.unique(configs_arr, return_counts=True)
                print("  По конфигурациям:")
                for config, count in zip(unique_configs, config_counts):
                    percentage = count / len(data) * 100
                    print(f"    {config}: {count} ({percentage:.1f}%)")

                # По типам данных
                unique_types, type_counts = np.unique(types_arr, return_counts=True)
                print("  По типам данных:")
                for dtype, count in zip(unique_types, type_counts):
                    percentage = count / len(data) * 100
                    print(f"    {dtype}: {count} ({percentage:.1f}%)")

                # Комбинированное распределение
                print("  Комбинированное (конфигурация_тип):")
                combined_current = [f"{c}_{t}" for c, t in zip(configs_arr, types_arr)]
                unique_combined, combined_counts = np.unique(combined_current, return_counts=True)
                for combo, count in zip(unique_combined, combined_counts):
                    percentage = count / len(data) * 100
                    print(f"    {combo}: {count} ({percentage:.1f}%)")

            return (train_spectra, train_mu_a, train_mu_s), \
                (val_spectra, val_mu_a, val_mu_s), \
                (test_spectra, test_mu_a, test_mu_s)

        train_data, val_data, test_data = prepare_dataset()

        print(f"  Обучающая выборка: {train_data[0].shape[0]}")
        print(f"  Валидационная выборка: {val_data[0].shape[0]}")
        print(f"  Тестовая выборка: {test_data[0].shape[0]}")

        # 3. Предобработка
        print("\n3. Гибридная предобработка...")
        preprocessor = HybridPreprocessor(use_fast_transform=True)
        preprocessor.fit(*train_data)

        train_data_proc = preprocessor.transform(*train_data)
        val_data_proc = preprocessor.transform(*val_data)
        test_data_proc = preprocessor.transform(*test_data)

        # 4. Создание DataLoader
        print("\n4. Создание DataLoader...")
        batch_size = 64

        train_dataset = HybridDRSDataset(*train_data_proc, augment_prob=0.4, use_smart_aug=True)
        val_dataset = HybridDRSDataset(*val_data_proc, augment_prob=0.0)
        test_dataset = HybridDRSDataset(*test_data_proc, augment_prob=0.0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=torch.cuda.is_available())

        print(f"  Batch size: {batch_size}")
        print(f"  Используется pin_memory: {torch.cuda.is_available()}")

        # 5. Создание модели
        print("\n5. Создание гибридной модели...")
        model = HybridDRSCNN(input_channels=591, output_size=591, use_depthwise=use_depthwise)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Всего параметров: {total_params:,}")
        print(f"  Обучаемых параметров: {trainable_params:,}")

        # 6. Обучение
        print("\n6. Начало гибридного обучения...")
        train_losses, val_losses, train_losses_a, train_losses_s, val_losses_a, val_losses_s = train_hybrid_model(
            model, train_loader, val_loader, epochs=180, lr=0.001,
            use_mixed_precision=use_mixed_precision and torch.cuda.is_available()
        )

        # 7. Оценка
        print("\n7. Оценка гибридной модели...")
        metrics = evaluate_hybrid_model(model, test_loader, preprocessor)

        # 8. Сохранение результатов
        print("\n8. Сохранение модели и результатов...")

        torch.save({
            'epoch': len(train_losses),
            'model_state_dict': model.state_dict(),
            'preprocessor_config': {
                'scaler_spectra_mean': preprocessor.scaler_spectra.mean_,
                'scaler_spectra_scale': preprocessor.scaler_spectra.scale_,
                'scaler_mu_a_mean': preprocessor.scaler_mu_a.mean_,
                'scaler_mu_a_scale': preprocessor.scaler_mu_a.scale_,
                'scaler_mu_s_mean': preprocessor.scaler_mu_s.mean_,
                'scaler_mu_s_scale': preprocessor.scaler_mu_s.scale_,
                'mu_s_stats': preprocessor.mu_s_stats,
                'use_fast_transform': preprocessor.use_fast_transform
            },
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'model': 'HybridDRSCNN',
                'input_channels': 591,
                'output_size': 591,
                'use_depthwise': use_depthwise,
                'use_mixed_precision': use_mixed_precision,
                'improvements': [
                    'hybrid_architecture',
                    'efficient_attention',
                    'adaptive_loss',
                    'mixed_precision',
                    'smart_augmentation'
                ],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }, 'hybrid_model_complete.pth')

        print("Модель сохранена как 'hybrid_model_complete.pth'")

        # 9. Итоговый анализ
        print(f"\n" + "=" * 60)
        print("ИТОГОВЫЙ АНАЛИЗ ГИБРИДНОЙ МОДЕЛИ")
        print("=" * 60)

        print(f"\nРезультаты модели:")
        print(f"  R² μ_s: {metrics['r2_s']:.4f}")
        print(f"  R² μ_a: {metrics['r2_a']:.4f}")

        print(f"\nКлючевые характеристики модели:")
        print("  1. Гибридная архитектура с остаточными блоками")
        print("  2. Эффективный механизм внимания")
        print("  3. Адаптивная функция потерь")
        print("  4. Mixed precision для ускорения на GPU")
        print("  5. Умная аугментация для лучшей обобщающей способности")
        print("  6. Эффективные свертки с depthwise оптимизацией")

        # Визуализация результатов
        visualize_hybrid_results(train_losses, val_losses, metrics)

        return model, metrics, preprocessor

    except Exception as e:
        print(f"\nОшибка при выполнении: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ============================================================================
# 9. Функция визуализации (С ГРАФИКАМИ ДЛЯ μ_a И μ_s)
# ============================================================================

def visualize_hybrid_results(train_losses, val_losses, metrics, save_prefix='hybrid'):
    """
    Визуализация результатов гибридной модели с исправленными относительными ошибками
    """
    if not train_losses:
        print("Нет данных для визуализации")
        return

    # Создаем папку для графиков если она не существует
    plot_dir = "Plot_CNN"
    os.makedirs(plot_dir, exist_ok=True)

    # Первый график: Основные результаты
    plt.figure(figsize=(20, 12))

    # 1. История обучения
    plt.subplot(3, 4, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Val Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('История обучения гибридной модели')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Сравнение истинных и предсказанных значений μ_s
    plt.subplot(3, 4, 2)
    # Фильтруем очень большие значения для лучшей визуализации
    mask_s = (metrics['true_s'] < np.percentile(metrics['true_s'], 95)) & \
             (metrics['pred_s'] < np.percentile(metrics['pred_s'], 95))
    true_s_filtered = metrics['true_s'][mask_s].flatten()
    pred_s_filtered = metrics['pred_s'][mask_s].flatten()

    plt.scatter(true_s_filtered, pred_s_filtered,
                alpha=0.3, s=1, color='blue')
    lims = [min(true_s_filtered.min(), pred_s_filtered.min()),
            max(true_s_filtered.max(), pred_s_filtered.max())]
    plt.plot(lims, lims, 'r--', alpha=0.8, label='Идеальная линия', linewidth=2)
    plt.xlabel('Истинные значения μ_s')
    plt.ylabel('Предсказанные значения μ_s')
    plt.title(f'R² μ_s = {metrics["r2_s"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Сравнение истинных и предсказанных значений μ_a
    plt.subplot(3, 4, 3)
    # Фильтруем очень большие значения для лучшей визуализации
    mask_a = (metrics['true_a'] < np.percentile(metrics['true_a'], 95)) & \
             (metrics['pred_a'] < np.percentile(metrics['pred_a'], 95))
    true_a_filtered = metrics['true_a'][mask_a].flatten()
    pred_a_filtered = metrics['pred_a'][mask_a].flatten()

    plt.scatter(true_a_filtered, pred_a_filtered,
                alpha=0.3, s=1, color='green')
    lims = [min(true_a_filtered.min(), pred_a_filtered.min()),
            max(true_a_filtered.max(), pred_a_filtered.max())]
    plt.plot(lims, lims, 'r--', alpha=0.8, label='Идеальная линия', linewidth=2)
    plt.xlabel('Истинные значения μ_a')
    plt.ylabel('Предсказанные значения μ_a')
    plt.title(f'R² μ_a = {metrics["r2_a"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Распределение ошибок для μ_s
    plt.subplot(3, 4, 4)
    errors_s = metrics['pred_s'] - metrics['true_s']
    # Исключаем выбросы для визуализации
    errors_s_filtered = errors_s[(np.abs(errors_s) < np.percentile(np.abs(errors_s), 95))]
    plt.hist(errors_s_filtered.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black', label='μ_s')
    plt.xlabel('Ошибка предсказания')
    plt.ylabel('Частота')
    plt.title(f'Распределение ошибок μ_s (σ={errors_s.std():.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Распределение ошибок для μ_a
    plt.subplot(3, 4, 5)
    errors_a = metrics['pred_a'] - metrics['true_a']
    # Исключаем выбросы для визуализации
    errors_a_filtered = errors_a[(np.abs(errors_a) < np.percentile(np.abs(errors_a), 95))]
    plt.hist(errors_a_filtered.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black', label='μ_a')
    plt.xlabel('Ошибка предсказания')
    plt.ylabel('Частота')
    plt.title(f'Распределение ошибок μ_a (σ={errors_a.std():.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Метрики качества для μ_s
    plt.subplot(3, 4, 6)
    metrics_names = ['R²', 'RMSE', 'MAE', 'sMAPE%']
    mu_s_metrics = [metrics['r2_s'], metrics['rmse_s'], metrics['mae_s'], metrics['median_rel_error_s']]
    colors = ['blue', 'orange', 'red', 'purple']

    x = np.arange(len(metrics_names))
    bars_s = plt.bar(x, mu_s_metrics, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Значение')
    plt.title('Метрики качества для μ_s')
    plt.xticks(x, metrics_names)
    plt.grid(True, alpha=0.3, axis='y')

    for i, (bar, value) in enumerate(zip(bars_s, mu_s_metrics)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    # 7. Метрики качества для μ_a
    plt.subplot(3, 4, 7)
    mu_a_metrics = [metrics['r2_a'], metrics['rmse_a'], metrics['mae_a'], metrics['median_rel_error_a']]
    colors = ['green', 'orange', 'red', 'purple']

    x = np.arange(len(metrics_names))
    bars_a = plt.bar(x, mu_a_metrics, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Значение')
    plt.title('Метрики качества для μ_a')
    plt.xticks(x, metrics_names)
    plt.grid(True, alpha=0.3, axis='y')

    for i, (bar, value) in enumerate(zip(bars_a, mu_a_metrics)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    # 8. Логарифмическая относительная ошибка для μ_s (более устойчивая)
    plt.subplot(3, 4, 8)
    # Используем логарифмическую ошибку для стабильности
    log_error_s = np.log10(np.abs(metrics['pred_s'] - metrics['true_s']) + 1e-8)
    plt.hist(log_error_s.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('log10(Абсолютная ошибка + ε)')
    plt.ylabel('Частота')
    plt.title('Логарифмическое распределение ошибок μ_s')
    plt.grid(True, alpha=0.3)

    # 9. Логарифмическая относительная ошибка для μ_a
    plt.subplot(3, 4, 9)
    log_error_a = np.log10(np.abs(metrics['pred_a'] - metrics['true_a']) + 1e-8)
    plt.hist(log_error_a.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('log10(Абсолютная ошибка + ε)')
    plt.ylabel('Частота')
    plt.title('Логарифмическое распределение ошибок μ_a')
    plt.grid(True, alpha=0.3)

    # 10. Симметричная относительная ошибка по длинам волн для μ_s (ограниченная)
    plt.subplot(3, 4, 10)
    # Используем симметричную относительную ошибку и ограничиваем её
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors_s = 2 * np.abs(metrics['pred_s'] - metrics['true_s']) / \
                       (np.abs(metrics['pred_s']) + np.abs(metrics['true_s']) + 1e-8)
        rel_errors_s = np.clip(rel_errors_s, 0, 1)  # Ограничиваем 100%
        rel_errors_point_s = np.median(rel_errors_s, axis=0) * 100

    plt.plot(rel_errors_point_s, color='blue', linewidth=2)
    plt.xlabel('Индекс длины волны')
    plt.ylabel('Медианная sMAPE (%)')
    plt.title('Симметричная относительная ошибка μ_s')
    plt.ylim([0, 100])  # Ограничиваем до 100%
    plt.grid(True, alpha=0.3)

    # 11. Симметричная относительная ошибка по длинам волн для μ_a (ограниченная)
    plt.subplot(3, 4, 11)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors_a = 2 * np.abs(metrics['pred_a'] - metrics['true_a']) / \
                       (np.abs(metrics['pred_a']) + np.abs(metrics['true_a']) + 1e-8)
        rel_errors_a = np.clip(rel_errors_a, 0, 1)  # Ограничиваем 100%
        rel_errors_point_a = np.median(rel_errors_a, axis=0) * 100

    plt.plot(rel_errors_point_a, color='green', linewidth=2)
    plt.xlabel('Индекс длины волны')
    plt.ylabel('Медианная sMAPE (%)')
    plt.title('Симметричная относительная ошибка μ_a')
    plt.ylim([0, 100])  # Ограничиваем до 100%
    plt.grid(True, alpha=0.3)

    # 12. Сравнение медианных sMAPE ошибок
    plt.subplot(3, 4, 12)
    labels = ['μ_a', 'μ_s']
    rel_errors_median = [metrics['median_rel_error_a'], metrics['median_rel_error_s']]
    colors = ['green', 'blue']

    plt.bar(labels, rel_errors_median, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Медианная sMAPE (%)')
    plt.title('Сравнение медианных симметричных ошибок')
    plt.grid(True, alpha=0.3, axis='y')

    for i, (label, value) in enumerate(zip(labels, rel_errors_median)):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Сохраняем график
    plot_path = os.path.join(plot_dir, f'{save_prefix}_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"График сохранен в: {plot_path}")
# ============================================================================
# Запуск программы
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ГИБРИДНАЯ CNN МОДЕЛЬ - ОПТИМАЛЬНОЕ РЕШЕНИЕ")
    print("=" * 60)

    print("\nВыберите режим работы:")
    print("1. Полное обучение гибридной модели")
    print("2. Быстрая проверка (демо)")
    print("3. Обучение без depthwise сверток")

    choice = input("\nВведите номер (1-3): ").strip()

    if choice == "1":
        print("\nЗапуск полного обучения гибридной модели...")
        model, metrics, preprocessor = run_hybrid_training(
            use_depthwise=True,
            use_mixed_precision=True
        )

        if metrics:
            print("\nОбучение успешно завершено!")

    elif choice == "2":
        print("\nДемо-режим: создание гибридной модели...")
        model = HybridDRSCNN()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Параметров модели: {total_params:,}")
        print("Гибридная модель создана успешно!")

    elif choice == "3":
        print("\nОбучение без depthwise сверток...")
        model, metrics, preprocessor = run_hybrid_training(
            use_depthwise=False,
            use_mixed_precision=True
        )

    else:
        print("\nНеверный выбор. Запуск полного обучения...")
        model, metrics, preprocessor = run_hybrid_training()