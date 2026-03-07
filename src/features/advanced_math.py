"""
Matemáticas Avanzadas para Detección de Patrones Ocultos.

Incluye:
- Wavelets (análisis multi-escala)
- Información Mutua (dependencias no lineales)
- PCA/ICA (componentes ocultos)
- Recurrence Plots (patrones repetitivos)
- Copulas (correlaciones de cola)
- Persistent Homology (topología)
- Gramian Angular Fields (imágenes de series)

Estos métodos detectan patrones que los indicadores tradicionales NO pueden ver.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# WAVELETS: ANÁLISIS MULTI-ESCALA
# =============================================================================

def haar_wavelet_transform(data: np.ndarray, levels: int = 4) -> Dict[str, np.ndarray]:
    """
    Transformada Wavelet de Haar.
    
    Descompone la señal en diferentes escalas temporales simultáneamente.
    
    En matemáticas: f(t) = Σ c_j,k × ψ_j,k(t)
    En trading: Ver tendencias de corto, medio y largo plazo AL MISMO TIEMPO
    
    Interpretación:
    - Nivel 1: Ruido de alta frecuencia (minutos)
    - Nivel 2: Oscilaciones cortas (horas)
    - Nivel 3: Oscilaciones medias (días)
    - Nivel 4: Tendencia de fondo (semanas)
    """
    result = {}
    
    # Asegurar longitud potencia de 2
    n = len(data)
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    padded = np.pad(data, (0, n_padded - n), mode='edge')
    
    approximation = padded.copy()
    
    for level in range(1, levels + 1):
        length = len(approximation)
        if length < 2:
            break
            
        # Haar transform
        half = length // 2
        
        # Coeficientes de aproximación (promedio)
        approx_new = (approximation[::2] + approximation[1::2]) / np.sqrt(2)
        
        # Coeficientes de detalle (diferencia)
        detail = (approximation[::2] - approximation[1::2]) / np.sqrt(2)
        
        # Interpolar de vuelta al tamaño original
        detail_full = np.interp(
            np.linspace(0, len(detail) - 1, n),
            np.arange(len(detail)),
            detail
        )
        
        approx_full = np.interp(
            np.linspace(0, len(approx_new) - 1, n),
            np.arange(len(approx_new)),
            approx_new
        )
        
        result[f'wavelet_detail_{level}'] = detail_full
        result[f'wavelet_approx_{level}'] = approx_full
        
        # Energía del nivel
        result[f'wavelet_energy_{level}'] = np.convolve(
            detail_full ** 2, 
            np.ones(20) / 20, 
            mode='same'
        )
        
        approximation = approx_new
    
    return result


def wavelet_variance_ratio(data: np.ndarray, levels: int = 4) -> Dict[str, np.ndarray]:
    """
    Ratio de varianza entre niveles wavelet.
    
    Indica qué escala temporal domina el movimiento actual.
    
    Interpretación:
    - Alto ratio nivel 1 → Ruido domina (evitar operar)
    - Alto ratio nivel 3-4 → Tendencia domina (seguir tendencia)
    """
    wavelets = haar_wavelet_transform(data, levels)
    
    result = {}
    
    # Varianzas por nivel
    variances = {}
    for level in range(1, levels + 1):
        detail = wavelets.get(f'wavelet_detail_{level}', np.zeros(len(data)))
        variances[level] = pd.Series(detail ** 2).rolling(50).mean().values
    
    total_var = sum(variances.values())
    
    # Ratios
    for level in range(1, levels + 1):
        result[f'wavelet_var_ratio_{level}'] = variances[level] / (total_var + 1e-10)
    
    # Dominancia: qué nivel tiene más varianza
    var_matrix = np.column_stack([variances[l] for l in range(1, levels + 1)])
    result['wavelet_dominant_level'] = np.argmax(var_matrix, axis=1) + 1
    
    return result


def calc_wavelet_coherence(prices: np.ndarray, volume: np.ndarray, levels: int = 3) -> Dict[str, np.ndarray]:
    """
    Coherencia Wavelet entre precio y volumen.
    
    Mide si precio y volumen están "sincronizados" en cada escala temporal.
    
    Interpretación:
    - Alta coherencia → Movimiento confirmado por volumen (confiable)
    - Baja coherencia → Movimiento sin volumen (sospechoso)
    """
    price_wavelets = haar_wavelet_transform(prices, levels)
    vol_wavelets = haar_wavelet_transform(volume, levels)
    
    result = {}
    
    for level in range(1, levels + 1):
        p_detail = price_wavelets.get(f'wavelet_detail_{level}', np.zeros(len(prices)))
        v_detail = vol_wavelets.get(f'wavelet_detail_{level}', np.zeros(len(volume)))
        
        # Coherencia = correlación rolling
        coherence = np.zeros(len(prices))
        window = 30
        
        for i in range(window, len(prices)):
            corr = np.corrcoef(p_detail[i-window:i], v_detail[i-window:i])[0, 1]
            coherence[i] = corr if not np.isnan(corr) else 0
        
        result[f'wavelet_coherence_{level}'] = coherence
    
    return result


# =============================================================================
# INFORMACIÓN MUTUA: DEPENDENCIAS NO LINEALES
# =============================================================================

def calc_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Información Mutua entre dos variables.
    
    I(X;Y) = Σ p(x,y) × log(p(x,y) / (p(x)p(y)))
    
    Mide TODA la dependencia (lineal y no lineal), no solo correlación.
    
    Interpretación:
    - MI = 0 → Variables independientes
    - MI alto → Variables muy dependientes (aunque correlación sea 0)
    """
    # Histograma 2D
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    
    # Probabilidades conjuntas
    p_xy = hist_2d / hist_2d.sum()
    
    # Probabilidades marginales
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # Información mutua
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def calc_transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = 10) -> float:
    """
    Transfer Entropy: Mide flujo de información de source → target.
    
    TE(X→Y) = I(Y_t; X_{t-lag} | Y_{t-1})
    
    Detecta CAUSALIDAD, no solo correlación.
    
    En trading: ¿El volumen CAUSA cambios en precio, o viceversa?
    """
    n = len(source)
    
    # Crear vectores con lag
    y_t = target[lag:]
    y_t1 = target[lag-1:-1] if lag > 1 else target[:-lag]
    x_lag = source[:-lag]
    
    # Discretizar
    y_t_d = np.digitize(y_t, np.linspace(y_t.min(), y_t.max(), bins))
    y_t1_d = np.digitize(y_t1, np.linspace(y_t1.min(), y_t1.max(), bins))
    x_lag_d = np.digitize(x_lag, np.linspace(x_lag.min(), x_lag.max(), bins))
    
    # Calcular entropías
    def entropy_from_counts(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    # H(Y_t | Y_{t-1})
    joint_yy = np.histogram2d(y_t_d, y_t1_d, bins=bins)[0]
    h_y_given_y = entropy_from_counts(joint_yy.flatten()) - entropy_from_counts(joint_yy.sum(axis=0))
    
    # H(Y_t | Y_{t-1}, X_{t-lag})
    # Aproximación: usar histograma 3D
    h_y_given_xy = 0
    for i in range(bins):
        mask = x_lag_d == i
        if mask.sum() > 10:
            sub_joint = np.histogram2d(y_t_d[mask], y_t1_d[mask], bins=bins)[0]
            if sub_joint.sum() > 0:
                h_y_given_xy += (mask.sum() / len(mask)) * entropy_from_counts(sub_joint.flatten())
    
    te = max(0, h_y_given_y - h_y_given_xy)
    
    return te


def calc_mi_features(df: pd.DataFrame, target_col: str = 'close', window: int = 100) -> Dict[str, np.ndarray]:
    """
    Calcula Información Mutua entre precio y otras variables.
    """
    result = {}
    prices = df[target_col].values
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    # MI con volumen
    if 'volume' in df.columns:
        volume = df['volume'].values
        mi_volume = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            mi_volume[i] = calc_mutual_information(
                returns[i-window:i],
                volume[i-window:i]
            )
        
        result['mi_volume'] = mi_volume
        
        # Transfer entropy: volumen → precio
        te_vol_price = np.zeros(len(prices))
        for i in range(window, len(prices)):
            te_vol_price[i] = calc_transfer_entropy(
                volume[i-window:i],
                prices[i-window:i],
                lag=1
            )
        result['te_volume_to_price'] = te_vol_price
    
    # MI con retornos pasados (auto-información)
    mi_auto = np.zeros(len(prices))
    for i in range(window, len(prices)):
        mi_auto[i] = calc_mutual_information(
            returns[i-window:i-1],
            returns[i-window+1:i]
        )
    result['mi_auto'] = mi_auto
    
    return result


# =============================================================================
# PCA / ICA: COMPONENTES OCULTOS
# =============================================================================

def calc_pca_features(df: pd.DataFrame, n_components: int = 5, window: int = 50) -> Dict[str, np.ndarray]:
    """
    PCA rolling: Extrae componentes principales de múltiples indicadores.
    
    Encuentra las "fuerzas ocultas" que mueven el mercado.
    
    Interpretación:
    - PC1 generalmente captura la tendencia principal
    - PC2-3 capturan oscilaciones secundarias
    - Varianza explicada indica qué tan predecible es el mercado
    """
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
    feature_cols = [c for c in numeric_cols if c not in exclude][:20]  # Limitar a 20
    
    if len(feature_cols) < n_components:
        return {}
    
    data = df[feature_cols].values
    n = len(data)
    
    result = {f'pca_{i+1}': np.zeros(n) for i in range(n_components)}
    result['pca_explained_var'] = np.zeros(n)
    result['pca_condition_number'] = np.zeros(n)
    
    scaler = StandardScaler()
    
    for i in range(window, n):
        segment = data[i-window:i]
        
        # Limpiar NaN
        segment = np.nan_to_num(segment, 0)
        
        try:
            # Normalizar
            segment_scaled = scaler.fit_transform(segment)
            
            # PCA
            pca = PCA(n_components=min(n_components, segment_scaled.shape[1]))
            transformed = pca.fit_transform(segment_scaled)
            
            # Último valor de cada componente
            for j in range(min(n_components, transformed.shape[1])):
                result[f'pca_{j+1}'][i] = transformed[-1, j]
            
            # Varianza explicada total
            result['pca_explained_var'][i] = sum(pca.explained_variance_ratio_[:3])
            
            # Condition number (indica estabilidad)
            eigenvalues = pca.explained_variance_
            if eigenvalues[-1] > 1e-10:
                result['pca_condition_number'][i] = eigenvalues[0] / eigenvalues[-1]
            
        except Exception:
            pass
    
    return result


def calc_ica_features(df: pd.DataFrame, n_components: int = 3, window: int = 50) -> Dict[str, np.ndarray]:
    """
    ICA (Independent Component Analysis): Encuentra señales INDEPENDIENTES.
    
    A diferencia de PCA, ICA busca componentes estadísticamente independientes,
    no solo ortogonales.
    
    En trading: Separa diferentes "fuentes" de movimiento del mercado.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
    feature_cols = [c for c in numeric_cols if c not in exclude][:15]
    
    if len(feature_cols) < n_components:
        return {}
    
    data = df[feature_cols].values
    n = len(data)
    
    result = {f'ica_{i+1}': np.zeros(n) for i in range(n_components)}
    result['ica_kurtosis'] = np.zeros(n)
    
    scaler = StandardScaler()
    
    for i in range(window, n):
        segment = data[i-window:i]
        segment = np.nan_to_num(segment, 0)
        
        try:
            segment_scaled = scaler.fit_transform(segment)
            
            ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
            transformed = ica.fit_transform(segment_scaled)
            
            for j in range(n_components):
                result[f'ica_{j+1}'][i] = transformed[-1, j]
            
            # Kurtosis promedio (indica no-gaussianidad)
            result['ica_kurtosis'][i] = np.mean([stats.kurtosis(transformed[:, j]) for j in range(n_components)])
            
        except Exception:
            pass
    
    return result


# =============================================================================
# RECURRENCE PLOTS: PATRONES REPETITIVOS
# =============================================================================

def calc_recurrence_features(prices: np.ndarray, embedding_dim: int = 3, delay: int = 1, 
                              threshold_pct: float = 10, window: int = 100) -> Dict[str, np.ndarray]:
    """
    Recurrence Quantification Analysis (RQA).
    
    Detecta cuándo el mercado está en un estado SIMILAR a estados pasados.
    
    En matemáticas: R(i,j) = 1 si ||x_i - x_j|| < ε
    En trading: ¿El patrón actual se parece a patrones pasados?
    
    Interpretación:
    - Alta recurrencia → Mercado en estado conocido (predecible)
    - Baja recurrencia → Mercado en territorio nuevo (cuidado)
    - Determinismo alto → Patrones claros
    """
    n = len(prices)
    
    result = {
        'rqa_recurrence_rate': np.zeros(n),
        'rqa_determinism': np.zeros(n),
        'rqa_laminarity': np.zeros(n),
        'rqa_entropy': np.zeros(n),
        'rqa_trapping_time': np.zeros(n)
    }
    
    for i in range(window, n):
        segment = prices[i-window:i]
        
        # Normalizar
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
        
        # Embedding (reconstrucción del espacio de fases)
        embedded = np.array([
            segment[j:j + (embedding_dim - 1) * delay + 1:delay]
            for j in range(len(segment) - (embedding_dim - 1) * delay)
        ])
        
        if len(embedded) < 10:
            continue
        
        # Matriz de distancias
        distances = squareform(pdist(embedded))
        
        # Threshold
        threshold = np.percentile(distances, threshold_pct)
        
        # Matriz de recurrencia
        R = (distances < threshold).astype(int)
        np.fill_diagonal(R, 0)  # Excluir diagonal
        
        n_points = len(R)
        
        # Recurrence Rate (RR)
        result['rqa_recurrence_rate'][i] = R.sum() / (n_points * (n_points - 1))
        
        # Determinismo (DET): Proporción de puntos en líneas diagonales
        diagonal_lines = []
        for k in range(-n_points + 2, n_points - 1):
            diag = np.diag(R, k)
            line_length = 0
            for val in diag:
                if val == 1:
                    line_length += 1
                else:
                    if line_length >= 2:
                        diagonal_lines.append(line_length)
                    line_length = 0
            if line_length >= 2:
                diagonal_lines.append(line_length)
        
        if diagonal_lines and R.sum() > 0:
            result['rqa_determinism'][i] = sum(diagonal_lines) / R.sum()
        
        # Laminarity (LAM): Proporción de puntos en líneas verticales
        vertical_lines = []
        for col in range(n_points):
            line_length = 0
            for row in range(n_points):
                if R[row, col] == 1:
                    line_length += 1
                else:
                    if line_length >= 2:
                        vertical_lines.append(line_length)
                    line_length = 0
            if line_length >= 2:
                vertical_lines.append(line_length)
        
        if vertical_lines and R.sum() > 0:
            result['rqa_laminarity'][i] = sum(vertical_lines) / R.sum()
        
        # Trapping Time (TT): Longitud promedio de líneas verticales
        if vertical_lines:
            result['rqa_trapping_time'][i] = np.mean(vertical_lines)
        
        # Entropía de la distribución de longitudes de línea
        if diagonal_lines:
            hist, _ = np.histogram(diagonal_lines, bins=10, density=True)
            hist = hist[hist > 0]
            result['rqa_entropy'][i] = -np.sum(hist * np.log2(hist + 1e-10))
    
    return result


# =============================================================================
# GRAMIAN ANGULAR FIELDS: SERIES A IMÁGENES
# =============================================================================

def calc_gaf_features(prices: np.ndarray, window: int = 20, output_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Gramian Angular Field (GAF).
    
    Convierte series temporales en "imágenes" que capturan correlaciones temporales.
    
    En matemáticas: GAF(i,j) = cos(φ_i + φ_j) donde φ = arccos(x̃)
    En trading: Representa relaciones entre puntos temporales como ángulos
    
    Útil para detectar patrones visuales que CNNs pueden aprender.
    """
    n = len(prices)
    
    # Features: estadísticas del GAF
    result = {
        'gaf_mean': np.zeros(n),
        'gaf_std': np.zeros(n),
        'gaf_trace': np.zeros(n),
        'gaf_det': np.zeros(n),
        'gaf_symmetry': np.zeros(n)
    }
    
    for i in range(window, n):
        segment = prices[i-window:i]
        
        # Normalizar a [-1, 1]
        min_val, max_val = segment.min(), segment.max()
        if max_val - min_val > 1e-10:
            normalized = 2 * (segment - min_val) / (max_val - min_val) - 1
        else:
            normalized = np.zeros_like(segment)
        
        # Clip para evitar errores en arccos
        normalized = np.clip(normalized, -1, 1)
        
        # Ángulos
        phi = np.arccos(normalized)
        
        # GASF (Gramian Angular Summation Field)
        gasf = np.outer(np.cos(phi), np.cos(phi)) + np.outer(np.sin(phi), np.sin(phi))
        
        # Reducir tamaño si es necesario
        if len(gasf) > output_size:
            step = len(gasf) // output_size
            gasf = gasf[::step, ::step][:output_size, :output_size]
        
        # Extraer features
        result['gaf_mean'][i] = gasf.mean()
        result['gaf_std'][i] = gasf.std()
        result['gaf_trace'][i] = np.trace(gasf) / len(gasf)
        
        # Determinante (escalar para estabilidad)
        try:
            det = np.linalg.det(gasf)
            result['gaf_det'][i] = np.sign(det) * np.log1p(np.abs(det))
        except:
            pass
        
        # Simetría
        result['gaf_symmetry'][i] = np.mean(np.abs(gasf - gasf.T))
    
    return result


# =============================================================================
# COPULAS: CORRELACIONES DE COLA
# =============================================================================

def calc_tail_dependence(x: np.ndarray, y: np.ndarray, quantile: float = 0.05) -> Tuple[float, float]:
    """
    Dependencia de Cola: ¿Las variables se mueven juntas en EXTREMOS?
    
    La correlación normal no captura esto. Dos variables pueden tener
    correlación = 0 pero moverse juntas en crashes.
    
    Interpretación:
    - Alta dependencia de cola inferior → Caen juntas en crashes
    - Alta dependencia de cola superior → Suben juntas en rallies
    """
    n = len(x)
    
    # Convertir a ranks uniformes
    x_rank = stats.rankdata(x) / (n + 1)
    y_rank = stats.rankdata(y) / (n + 1)
    
    # Dependencia de cola inferior
    lower_mask = (x_rank <= quantile) & (y_rank <= quantile)
    lower_dep = lower_mask.sum() / (n * quantile) if n * quantile > 0 else 0
    
    # Dependencia de cola superior
    upper_mask = (x_rank >= 1 - quantile) & (y_rank >= 1 - quantile)
    upper_dep = upper_mask.sum() / (n * quantile) if n * quantile > 0 else 0
    
    return lower_dep, upper_dep


def calc_copula_features(df: pd.DataFrame, window: int = 100) -> Dict[str, np.ndarray]:
    """
    Features basadas en Copulas.
    """
    n = len(df)
    prices = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(n)
    
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    vol_change = np.diff(volume) / (volume[:-1] + 1e-10)
    vol_change = np.insert(vol_change, 0, 0)
    
    result = {
        'tail_dep_lower': np.zeros(n),
        'tail_dep_upper': np.zeros(n),
        'tail_asymmetry': np.zeros(n),
        'kendall_tau': np.zeros(n)
    }
    
    for i in range(window, n):
        ret_seg = returns[i-window:i]
        vol_seg = vol_change[i-window:i]
        
        # Dependencia de cola
        lower, upper = calc_tail_dependence(ret_seg, vol_seg)
        result['tail_dep_lower'][i] = lower
        result['tail_dep_upper'][i] = upper
        result['tail_asymmetry'][i] = upper - lower
        
        # Kendall's Tau (correlación de rangos, más robusta)
        tau, _ = stats.kendalltau(ret_seg, vol_seg)
        result['kendall_tau'][i] = tau if not np.isnan(tau) else 0
    
    return result


# =============================================================================
# FEATURES DE COMPLEJIDAD
# =============================================================================

def calc_complexity_features(prices: np.ndarray, window: int = 50) -> Dict[str, np.ndarray]:
    """
    Features de complejidad de la serie temporal.
    """
    n = len(prices)
    
    result = {
        'complexity': np.zeros(n),
        'permutation_entropy': np.zeros(n),
        'approximate_entropy': np.zeros(n)
    }
    
    for i in range(window, n):
        segment = prices[i-window:i]
        returns = np.diff(segment) / (segment[:-1] + 1e-10)
        
        # Complejidad de Lempel-Ziv (simplificada)
        binary = ''.join(['1' if r > 0 else '0' for r in returns])
        
        # Contar patrones únicos
        patterns = set()
        for length in range(1, min(10, len(binary))):
            for j in range(len(binary) - length + 1):
                patterns.add(binary[j:j+length])
        
        result['complexity'][i] = len(patterns) / len(binary)
        
        # Permutation Entropy
        order = 3
        perms = []
        for j in range(len(returns) - order + 1):
            perm = tuple(np.argsort(returns[j:j+order]))
            perms.append(perm)
        
        if perms:
            unique, counts = np.unique(perms, axis=0, return_counts=True)
            probs = counts / counts.sum()
            result['permutation_entropy'][i] = -np.sum(probs * np.log2(probs + 1e-10))
    
    return result


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def calculate_advanced_math_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features de matemáticas avanzadas.
    """
    logger.info("🧮 Calculando features de MATEMÁTICAS AVANZADAS...")
    
    df = df.copy()
    prices = df['close'].values
    
    # 1. Wavelets
    logger.info("   Wavelets...")
    for k, v in haar_wavelet_transform(prices).items():
        df[k] = v
    for k, v in wavelet_variance_ratio(prices).items():
        df[k] = v
    if 'volume' in df.columns:
        for k, v in calc_wavelet_coherence(prices, df['volume'].values).items():
            df[k] = v
    
    # 2. Información Mutua
    logger.info("   Información Mutua...")
    for k, v in calc_mi_features(df).items():
        df[k] = v
    
    # 3. PCA/ICA
    logger.info("   PCA/ICA...")
    for k, v in calc_pca_features(df).items():
        df[k] = v
    for k, v in calc_ica_features(df).items():
        df[k] = v
    
    # 4. Recurrence
    logger.info("   Recurrence Plots...")
    for k, v in calc_recurrence_features(prices).items():
        df[k] = v
    
    # 5. GAF
    logger.info("   Gramian Angular Fields...")
    for k, v in calc_gaf_features(prices).items():
        df[k] = v
    
    # 6. Copulas
    logger.info("   Copulas...")
    for k, v in calc_copula_features(df).items():
        df[k] = v
    
    # 7. Complejidad
    logger.info("   Complejidad...")
    for k, v in calc_complexity_features(prices).items():
        df[k] = v
    
    # Limpiar
    df = df.ffill().bfill()
    
    # Contar
    math_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    logger.info(f"✅ {len(math_cols)} features de matemáticas avanzadas calculadas")
    
    return df


def get_advanced_math_columns() -> List[str]:
    """Retorna lista de columnas de features matemáticas."""
    return [
        # Wavelets
        'wavelet_detail_1', 'wavelet_detail_2', 'wavelet_detail_3', 'wavelet_detail_4',
        'wavelet_approx_1', 'wavelet_approx_2', 'wavelet_approx_3', 'wavelet_approx_4',
        'wavelet_energy_1', 'wavelet_energy_2', 'wavelet_energy_3', 'wavelet_energy_4',
        'wavelet_var_ratio_1', 'wavelet_var_ratio_2', 'wavelet_var_ratio_3', 'wavelet_var_ratio_4',
        'wavelet_dominant_level',
        'wavelet_coherence_1', 'wavelet_coherence_2', 'wavelet_coherence_3',
        
        # Información Mutua
        'mi_volume', 'te_volume_to_price', 'mi_auto',
        
        # PCA/ICA
        'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
        'pca_explained_var', 'pca_condition_number',
        'ica_1', 'ica_2', 'ica_3', 'ica_kurtosis',
        
        # Recurrence
        'rqa_recurrence_rate', 'rqa_determinism', 'rqa_laminarity',
        'rqa_entropy', 'rqa_trapping_time',
        
        # GAF
        'gaf_mean', 'gaf_std', 'gaf_trace', 'gaf_det', 'gaf_symmetry',
        
        # Copulas
        'tail_dep_lower', 'tail_dep_upper', 'tail_asymmetry', 'kendall_tau',
        
        # Complejidad
        'complexity', 'permutation_entropy', 'approximate_entropy'
    ]


# =============================================================================
# TEST
# =============================================================================

def test_advanced_math():
    """Test de features matemáticas avanzadas."""
    print("🧪 Test Features de Matemáticas Avanzadas...")
    
    # Datos dummy
    np.random.seed(42)
    n = 500
    
    t = np.linspace(0, 10, n)
    trend = 50000 + t * 1000
    cycle = 2000 * np.sin(2 * np.pi * t / 2)
    noise = np.cumsum(np.random.randn(n) * 50)
    
    prices = trend + cycle + noise
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='4h'),
        'open': prices + np.random.randn(n) * 50,
        'high': prices + np.abs(np.random.randn(n)) * 100,
        'low': prices - np.abs(np.random.randn(n)) * 100,
        'close': prices,
        'volume': np.random.exponential(1000, n)
    })
    
    # Añadir algunas features para PCA/ICA
    df['rsi'] = 50 + np.random.randn(n) * 10
    df['macd'] = np.random.randn(n) * 100
    
    # Calcular features
    df_math = calculate_advanced_math_features(df)
    
    # Verificar
    math_cols = get_advanced_math_columns()
    available = [c for c in math_cols if c in df_math.columns]
    
    print(f"✅ Features solicitadas: {len(math_cols)}")
    print(f"✅ Features disponibles: {len(available)}")
    
    # Estadísticas
    print(f"\n📊 Estadísticas de muestra:")
    for col in available[:8]:
        val = df_math[col].dropna()
        if len(val) > 0:
            print(f"   {col}: mean={val.mean():.4f}, std={val.std():.4f}")
    
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_advanced_math()
