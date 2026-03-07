"""
Física Aplicada al Mercado Financiero.

El precio se comporta como un sistema físico:
- Tiene masa (volumen)
- Tiene velocidad (cambio de precio)
- Tiene aceleración (cambio de velocidad)
- Tiene energía (volatilidad × volumen)
- Tiene momentum (precio × volumen)
- Oscila como ondas (Fourier)
- Tiende al equilibrio (mean reversion)

Este módulo traduce conceptos de física a features de trading.
"""

import numpy as np
import pandas as pd
from scipy import signal, fft
from scipy.stats import entropy
from typing import Dict, List, Tuple
from loguru import logger


# =============================================================================
# MECÁNICA CLÁSICA: VELOCIDAD, ACELERACIÓN, MOMENTUM
# =============================================================================

def calc_velocity(prices: np.ndarray, periods: List[int] = [1, 5, 10, 20]) -> Dict[str, np.ndarray]:
    """
    Velocidad = Primera derivada del precio (dp/dt).
    
    En física: v = dx/dt
    En trading: v = (P_t - P_{t-n}) / n
    
    Interpretación:
    - Velocidad positiva → precio subiendo
    - Velocidad creciente → tendencia acelerando
    - Velocidad decreciente → tendencia perdiendo fuerza
    """
    result = {}
    
    for p in periods:
        velocity = np.zeros(len(prices))
        velocity[p:] = (prices[p:] - prices[:-p]) / p
        
        # Normalizar por precio para comparabilidad
        velocity_pct = velocity / (prices + 1e-10)
        
        result[f'velocity_{p}'] = velocity_pct
    
    return result


def calc_acceleration(prices: np.ndarray, periods: List[int] = [1, 5, 10]) -> Dict[str, np.ndarray]:
    """
    Aceleración = Segunda derivada del precio (d²p/dt²).
    
    En física: a = dv/dt = d²x/dt²
    En trading: Cambio en la velocidad
    
    Interpretación:
    - Aceleración positiva → tendencia fortaleciéndose
    - Aceleración negativa → tendencia debilitándose
    - Aceleración = 0 → movimiento uniforme
    """
    velocities = calc_velocity(prices, periods)
    result = {}
    
    for p in periods:
        vel = velocities[f'velocity_{p}']
        
        accel = np.zeros(len(prices))
        accel[1:] = np.diff(vel)
        
        result[f'acceleration_{p}'] = accel
    
    return result


def calc_jerk(prices: np.ndarray, period: int = 5) -> np.ndarray:
    """
    Jerk = Tercera derivada (d³p/dt³).
    
    En física: Cambio en la aceleración
    En trading: Detecta puntos de inflexión extremos
    
    Alto jerk = cambio brusco inminente
    """
    accel = calc_acceleration(prices, [period])[f'acceleration_{period}']
    
    jerk = np.zeros(len(prices))
    jerk[1:] = np.diff(accel)
    
    return jerk


def calc_momentum_physics(prices: np.ndarray, volume: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
    """
    Momentum físico = masa × velocidad = volumen × velocidad_precio.
    
    En física: p = m × v
    En trading: El volumen actúa como "masa" del movimiento
    
    Interpretación:
    - Alto momentum → movimiento con fuerza (difícil de parar)
    - Bajo momentum → movimiento débil (fácil reversión)
    - Divergencia precio/momentum → posible reversión
    """
    velocity = calc_velocity(prices, [period])[f'velocity_{period}']
    
    # Normalizar volumen
    vol_norm = volume / (pd.Series(volume).rolling(period).mean().values + 1e-10)
    
    # Momentum = masa (volumen) × velocidad
    momentum = vol_norm * velocity
    
    # Momentum suavizado
    momentum_smooth = pd.Series(momentum).rolling(period).mean().values
    
    # Impulso (cambio en momentum)
    impulse = np.zeros(len(prices))
    impulse[1:] = np.diff(momentum)
    
    return {
        'momentum_phys': momentum,
        'momentum_smooth': momentum_smooth,
        'impulse': impulse
    }


# =============================================================================
# ENERGÍA Y TRABAJO
# =============================================================================

def calc_kinetic_energy(prices: np.ndarray, volume: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
    """
    Energía Cinética = ½ × masa × velocidad².
    
    En física: KE = ½mv²
    En trading: Mide la "fuerza" del movimiento actual
    
    Interpretación:
    - Alta energía → movimiento fuerte, puede continuar
    - Energía decreciente → movimiento perdiendo fuerza
    - Pico de energía → posible agotamiento
    """
    velocity = calc_velocity(prices, [period])[f'velocity_{period}']
    
    # Masa = volumen normalizado
    vol_norm = volume / (pd.Series(volume).rolling(50).mean().values + 1e-10)
    
    # KE = ½mv²
    kinetic_energy = 0.5 * vol_norm * (velocity ** 2)
    
    # Energía acumulada (rolling)
    ke_cumulative = pd.Series(kinetic_energy).rolling(period).sum().values
    
    return {
        'kinetic_energy': kinetic_energy,
        'kinetic_energy_cumsum': ke_cumulative
    }


def calc_potential_energy(prices: np.ndarray, period: int = 50) -> Dict[str, np.ndarray]:
    """
    Energía Potencial = Distancia al equilibrio (media móvil).
    
    En física: PE = mgh (altura sobre referencia)
    En trading: Distancia del precio a su "nivel de equilibrio"
    
    Interpretación:
    - Alta PE positiva → precio muy arriba de equilibrio (puede caer)
    - Alta PE negativa → precio muy abajo de equilibrio (puede subir)
    - PE ≈ 0 → precio en equilibrio
    """
    # Equilibrio = media móvil
    equilibrium = pd.Series(prices).rolling(period).mean().values
    
    # Distancia al equilibrio (normalizada)
    displacement = (prices - equilibrium) / (equilibrium + 1e-10)
    
    # Energía potencial = ½kx² (como un resorte)
    # k = constante de "rigidez" del mercado
    k = 1.0 / (pd.Series(displacement).rolling(period).std().values + 1e-10)
    
    potential_energy = 0.5 * k * (displacement ** 2)
    
    # Signo indica dirección
    pe_signed = potential_energy * np.sign(displacement)
    
    return {
        'potential_energy': potential_energy,
        'potential_energy_signed': pe_signed,
        'displacement': displacement
    }


def calc_total_energy(prices: np.ndarray, volume: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
    """
    Energía Total = Cinética + Potencial.
    
    En física: E = KE + PE (conservación de energía)
    En trading: La energía total tiende a conservarse
    
    Interpretación:
    - Energía total constante → mercado en equilibrio dinámico
    - Energía total aumentando → fuerza externa (noticias, ballenas)
    - Energía total disminuyendo → mercado perdiendo interés
    """
    ke = calc_kinetic_energy(prices, volume, period)['kinetic_energy']
    pe = calc_potential_energy(prices, period)['potential_energy']
    
    total_energy = ke + pe
    
    # Cambio en energía total
    energy_change = np.zeros(len(prices))
    energy_change[1:] = np.diff(total_energy)
    
    # Ratio KE/PE (indica si energía es de movimiento o posición)
    ke_pe_ratio = ke / (pe + 1e-10)
    
    return {
        'total_energy': total_energy,
        'energy_change': energy_change,
        'ke_pe_ratio': ke_pe_ratio
    }


# =============================================================================
# OSCILACIONES Y ONDAS (Fourier)
# =============================================================================

def calc_fourier_features(prices: np.ndarray, n_components: int = 5) -> Dict[str, np.ndarray]:
    """
    Análisis de Fourier - Descompone el precio en ondas sinusoidales.
    
    En física: Cualquier señal = suma de senos y cosenos
    En trading: Detecta ciclos ocultos en el precio
    
    Interpretación:
    - Frecuencias dominantes → ciclos del mercado
    - Fase → en qué punto del ciclo estamos
    - Amplitud → fuerza del ciclo
    """
    result = {}
    window = 100  # Ventana de análisis
    
    dominant_freqs = np.zeros((len(prices), n_components))
    dominant_amplitudes = np.zeros((len(prices), n_components))
    dominant_phases = np.zeros((len(prices), n_components))
    
    for i in range(window, len(prices)):
        segment = prices[i-window:i]
        segment_detrended = segment - np.linspace(segment[0], segment[-1], window)
        
        # FFT
        fft_vals = np.fft.fft(segment_detrended)
        freqs = np.fft.fftfreq(window)
        
        # Solo frecuencias positivas
        pos_mask = freqs > 0
        fft_pos = np.abs(fft_vals[pos_mask])
        freqs_pos = freqs[pos_mask]
        phases_pos = np.angle(fft_vals[pos_mask])
        
        # Top n componentes
        top_idx = np.argsort(fft_pos)[-n_components:][::-1]
        
        dominant_freqs[i] = freqs_pos[top_idx]
        dominant_amplitudes[i] = fft_pos[top_idx] / window
        dominant_phases[i] = phases_pos[top_idx]
    
    # Features principales
    result['dominant_freq_1'] = dominant_freqs[:, 0]
    result['dominant_freq_2'] = dominant_freqs[:, 1]
    result['dominant_amplitude_1'] = dominant_amplitudes[:, 0]
    result['dominant_amplitude_2'] = dominant_amplitudes[:, 1]
    result['dominant_phase_1'] = dominant_phases[:, 0]
    result['dominant_phase_2'] = dominant_phases[:, 1]
    
    # Ciclo dominante en períodos
    result['dominant_period'] = 1.0 / (dominant_freqs[:, 0] + 1e-10)
    
    # Energía espectral total
    result['spectral_energy'] = np.sum(dominant_amplitudes ** 2, axis=1)
    
    # Concentración espectral (qué tan dominante es la frecuencia principal)
    total_amp = np.sum(dominant_amplitudes, axis=1) + 1e-10
    result['spectral_concentration'] = dominant_amplitudes[:, 0] / total_amp
    
    return result


def calc_hilbert_transform(prices: np.ndarray, period: int = 20) -> Dict[str, np.ndarray]:
    """
    Transformada de Hilbert - Extrae fase instantánea y amplitud envolvente.
    
    En física: Convierte señal real en analítica (compleja)
    En trading: Detecta la fase del ciclo actual
    
    Interpretación:
    - Fase 0° → inicio de ciclo alcista
    - Fase 90° → pico del ciclo
    - Fase 180° → inicio de ciclo bajista
    - Fase 270° → valle del ciclo
    """
    # Detrend primero
    prices_detrended = prices - pd.Series(prices).rolling(period*2).mean().values
    prices_detrended = np.nan_to_num(prices_detrended, 0)
    
    # Hilbert transform
    analytic_signal = signal.hilbert(prices_detrended)
    
    # Amplitud envolvente
    amplitude_envelope = np.abs(analytic_signal)
    
    # Fase instantánea
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Frecuencia instantánea
    instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi)
    instantaneous_freq = np.insert(instantaneous_freq, 0, 0)
    
    # Fase normalizada a [0, 2π]
    phase_normalized = instantaneous_phase % (2 * np.pi)
    
    # Indicador de posición en ciclo
    cycle_position = np.sin(phase_normalized)  # -1 a 1
    
    return {
        'hilbert_amplitude': amplitude_envelope,
        'hilbert_phase': phase_normalized,
        'hilbert_freq': instantaneous_freq,
        'cycle_position': cycle_position
    }


# =============================================================================
# ENTROPÍA Y CAOS
# =============================================================================

def calc_entropy_features(prices: np.ndarray, period: int = 50) -> Dict[str, np.ndarray]:
    """
    Entropía - Mide el desorden/incertidumbre del mercado.
    
    En física: S = -Σ p_i × log(p_i)
    En trading: Alta entropía = mercado impredecible
    
    Interpretación:
    - Alta entropía → mercado caótico, difícil predecir
    - Baja entropía → mercado ordenado, patrones claros
    - Entropía creciente → mercado volviéndose caótico
    - Entropía decreciente → patrón emergiendo
    """
    result = {}
    n_bins = 20
    
    shannon_entropy = np.zeros(len(prices))
    
    for i in range(period, len(prices)):
        returns = np.diff(prices[i-period:i]) / (prices[i-period:i-1] + 1e-10)
        
        # Histograma de retornos
        hist, _ = np.histogram(returns, bins=n_bins, density=True)
        hist = hist + 1e-10  # Evitar log(0)
        hist = hist / hist.sum()  # Normalizar
        
        # Entropía de Shannon
        shannon_entropy[i] = entropy(hist)
    
    # Normalizar por máxima entropía posible
    max_entropy = np.log(n_bins)
    shannon_entropy_norm = shannon_entropy / max_entropy
    
    result['shannon_entropy'] = shannon_entropy_norm
    
    # Cambio en entropía
    entropy_change = np.zeros(len(prices))
    entropy_change[1:] = np.diff(shannon_entropy_norm)
    result['entropy_change'] = entropy_change
    
    # Entropía de volumen
    return result


def calc_sample_entropy(prices: np.ndarray, m: int = 2, r_mult: float = 0.2, period: int = 50) -> np.ndarray:
    """
    Sample Entropy - Mide complejidad/regularidad de la serie.
    
    Más robusta que Shannon para series temporales.
    
    Interpretación:
    - Baja SampEn → serie regular, predecible
    - Alta SampEn → serie irregular, caótica
    """
    def _sample_entropy(ts, m, r):
        N = len(ts)
        if N < m + 1:
            return 0
        
        # Crear vectores de embedding
        def create_templates(data, m):
            return np.array([data[i:i+m] for i in range(len(data) - m + 1)])
        
        # Contar matches
        def count_matches(templates, r):
            count = 0
            N = len(templates)
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count
        
        templates_m = create_templates(ts, m)
        templates_m1 = create_templates(ts, m + 1)
        
        B = count_matches(templates_m, r)
        A = count_matches(templates_m1, r)
        
        if B == 0:
            return 0
        
        return -np.log(A / B) if A > 0 else 0
    
    result = np.zeros(len(prices))
    
    for i in range(period, len(prices)):
        segment = prices[i-period:i]
        r = r_mult * np.std(segment)
        
        # Normalizar segmento
        segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
        
        result[i] = _sample_entropy(segment_norm, m, r)
    
    return result


def calc_lyapunov_exponent(prices: np.ndarray, period: int = 50, tau: int = 1) -> np.ndarray:
    """
    Exponente de Lyapunov - Mide sensibilidad a condiciones iniciales (caos).
    
    En física: λ > 0 indica sistema caótico
    En trading: 
    - λ alto → mercado muy sensible, pequeños cambios causan grandes efectos
    - λ bajo → mercado estable, predecible
    """
    def _lyapunov(ts, tau=1):
        n = len(ts)
        if n < 10:
            return 0
        
        # Reconstrucción del espacio de fases
        m = 3  # dimensión de embedding
        vectors = np.array([ts[i:i+m*tau:tau] for i in range(n - m*tau)])
        
        if len(vectors) < 2:
            return 0
        
        # Encontrar vecinos cercanos y medir divergencia
        divergences = []
        
        for i in range(len(vectors) - 1):
            # Distancia al punto más cercano (que no sea él mismo ni adyacente)
            dists = np.linalg.norm(vectors - vectors[i], axis=1)
            dists[max(0, i-5):min(len(dists), i+6)] = np.inf  # Excluir vecinos temporales
            
            j = np.argmin(dists)
            initial_dist = dists[j]
            
            if initial_dist < 1e-10 or i + 1 >= len(vectors) or j + 1 >= len(vectors):
                continue
            
            # Divergencia después de un paso
            final_dist = np.linalg.norm(vectors[i+1] - vectors[j+1])
            
            if initial_dist > 0 and final_dist > 0:
                divergences.append(np.log(final_dist / initial_dist))
        
        return np.mean(divergences) if divergences else 0
    
    result = np.zeros(len(prices))
    
    for i in range(period, len(prices)):
        segment = prices[i-period:i]
        segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
        result[i] = _lyapunov(segment_norm, tau)
    
    return result


# =============================================================================
# FRACTALES
# =============================================================================

def calc_hurst_exponent(prices: np.ndarray, period: int = 100) -> np.ndarray:
    """
    Exponente de Hurst - Mide la "memoria" del mercado.
    
    En física: H describe comportamiento de movimiento browniano fraccional
    
    Interpretación:
    - H < 0.5 → Mean-reverting (precio tiende a volver a la media)
    - H = 0.5 → Random walk (impredecible)
    - H > 0.5 → Trending (tendencia persistente)
    
    Esto es CLAVE para elegir estrategia:
    - H < 0.5 → Usar estrategias de reversión a la media
    - H > 0.5 → Usar estrategias de seguimiento de tendencia
    """
    def _hurst(ts):
        n = len(ts)
        if n < 20:
            return 0.5
        
        # R/S análisis
        lags = range(2, min(n // 2, 50))
        rs_values = []
        
        for lag in lags:
            # Dividir en subseries
            n_subseries = n // lag
            rs_subseries = []
            
            for i in range(n_subseries):
                subseries = ts[i*lag:(i+1)*lag]
                
                # Desviación acumulada de la media
                mean = np.mean(subseries)
                cumsum = np.cumsum(subseries - mean)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subseries)
                
                if S > 0:
                    rs_subseries.append(R / S)
            
            if rs_subseries:
                rs_values.append((lag, np.mean(rs_subseries)))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Regresión log-log
        lags_log = np.log([x[0] for x in rs_values])
        rs_log = np.log([x[1] for x in rs_values])
        
        slope, _ = np.polyfit(lags_log, rs_log, 1)
        
        return np.clip(slope, 0, 1)
    
    result = np.zeros(len(prices))
    
    for i in range(period, len(prices)):
        returns = np.diff(np.log(prices[i-period:i] + 1e-10))
        result[i] = _hurst(returns)
    
    return result


def calc_fractal_dimension(prices: np.ndarray, period: int = 50) -> np.ndarray:
    """
    Dimensión Fractal - Mide la "rugosidad" del precio.
    
    En física: D describe cuánto "llena" el espacio una curva
    
    Interpretación:
    - D ≈ 1.0 → Precio suave, trending
    - D ≈ 1.5 → Precio moderadamente rugoso
    - D ≈ 2.0 → Precio muy rugoso, ruidoso
    
    Útil para:
    - Detectar consolidación (D alto)
    - Detectar tendencias limpias (D bajo)
    """
    def _fractal_dim(ts):
        n = len(ts)
        if n < 10:
            return 1.5
        
        # Método de Higuchi
        k_max = min(10, n // 4)
        lk = []
        
        for k in range(1, k_max + 1):
            # Calcular longitud para escala k
            L_k = []
            for m in range(1, k + 1):
                # Reconstruir subserie
                indices = range(m - 1, n, k)
                if len(indices) < 2:
                    continue
                
                subseries = ts[list(indices)]
                
                # Longitud normalizada
                L_m = np.sum(np.abs(np.diff(subseries))) * (n - 1) / (k * len(indices))
                L_k.append(L_m)
            
            if L_k:
                lk.append((k, np.mean(L_k)))
        
        if len(lk) < 2:
            return 1.5
        
        # Regresión log-log
        k_log = np.log([x[0] for x in lk])
        l_log = np.log([x[1] + 1e-10 for x in lk])
        
        slope, _ = np.polyfit(k_log, l_log, 1)
        
        return np.clip(-slope, 1, 2)
    
    result = np.zeros(len(prices))
    
    for i in range(period, len(prices)):
        segment = prices[i-period:i]
        segment_norm = (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-10)
        result[i] = _fractal_dim(segment_norm)
    
    return result


# =============================================================================
# MEAN REVERSION (Ley de Hooke)
# =============================================================================

def calc_mean_reversion(prices: np.ndarray, periods: List[int] = [20, 50, 100]) -> Dict[str, np.ndarray]:
    """
    Mean Reversion basado en Ley de Hooke: F = -kx
    
    En física: Un resorte estirado tiende a volver a su posición original
    En trading: El precio tiende a volver a su media
    
    Fuerza de reversión = -k × distancia_a_media
    
    Interpretación:
    - Alta fuerza positiva → precio muy arriba, probabilidad de caer
    - Alta fuerza negativa → precio muy abajo, probabilidad de subir
    """
    result = {}
    
    for period in periods:
        # Media móvil = posición de equilibrio
        equilibrium = pd.Series(prices).rolling(period).mean().values
        
        # Desplazamiento
        displacement = prices - equilibrium
        
        # Desplazamiento normalizado
        std = pd.Series(prices).rolling(period).std().values
        z_score = displacement / (std + 1e-10)
        
        # Constante k (rigidez) = qué tan rápido revierte
        # Estimada como correlación entre desplazamiento y retorno siguiente
        returns = np.zeros(len(prices))
        returns[1:] = np.diff(prices) / (prices[:-1] + 1e-10)
        
        # Fuerza de reversión = -k × z_score
        # (aproximamos k como la velocidad histórica de reversión)
        k = np.zeros(len(prices))
        for i in range(period * 2, len(prices)):
            corr = np.corrcoef(z_score[i-period:i], returns[i-period+1:i+1])[0, 1]
            k[i] = -corr if not np.isnan(corr) else 0
        
        reversion_force = k * z_score
        
        result[f'mr_zscore_{period}'] = z_score
        result[f'mr_force_{period}'] = reversion_force
        result[f'mr_k_{period}'] = k
    
    return result


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def calculate_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features de física.
    
    Args:
        df: DataFrame con OHLCV
        
    Returns:
        DataFrame con features de física añadidas
    """
    logger.info("⚛️ Calculando features de FÍSICA...")
    
    df = df.copy()
    prices = df['close'].values
    volume = df['volume'].values
    
    # 1. Mecánica: Velocidad, Aceleración
    logger.info("   Mecánica (velocidad, aceleración)...")
    for k, v in calc_velocity(prices).items():
        df[k] = v
    for k, v in calc_acceleration(prices).items():
        df[k] = v
    df['jerk'] = calc_jerk(prices)
    
    # 2. Momentum físico
    logger.info("   Momentum físico...")
    for k, v in calc_momentum_physics(prices, volume).items():
        df[k] = v
    
    # 3. Energía
    logger.info("   Energía (cinética, potencial)...")
    for k, v in calc_kinetic_energy(prices, volume).items():
        df[k] = v
    for k, v in calc_potential_energy(prices).items():
        df[k] = v
    for k, v in calc_total_energy(prices, volume).items():
        df[k] = v
    
    # 4. Ondas (Fourier)
    logger.info("   Ondas (Fourier, Hilbert)...")
    for k, v in calc_fourier_features(prices).items():
        df[k] = v
    for k, v in calc_hilbert_transform(prices).items():
        df[k] = v
    
    # 5. Entropía y Caos
    logger.info("   Entropía y Caos...")
    for k, v in calc_entropy_features(prices).items():
        df[k] = v
    df['sample_entropy'] = calc_sample_entropy(prices)
    df['lyapunov'] = calc_lyapunov_exponent(prices)
    
    # 6. Fractales
    logger.info("   Fractales (Hurst, dimensión)...")
    df['hurst'] = calc_hurst_exponent(prices)
    df['fractal_dim'] = calc_fractal_dimension(prices)
    
    # 7. Mean Reversion
    logger.info("   Mean Reversion...")
    for k, v in calc_mean_reversion(prices).items():
        df[k] = v
    
    # Limpiar NaN
    df = df.ffill().bfill()
    
    # Contar features
    physics_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    logger.info(f"✅ {len(physics_cols)} features de física calculadas")
    
    return df


def get_physics_feature_columns() -> List[str]:
    """Retorna lista de columnas de features de física."""
    return [
        # Mecánica
        'velocity_1', 'velocity_5', 'velocity_10', 'velocity_20',
        'acceleration_1', 'acceleration_5', 'acceleration_10',
        'jerk',
        
        # Momentum
        'momentum_phys', 'momentum_smooth', 'impulse',
        
        # Energía
        'kinetic_energy', 'kinetic_energy_cumsum',
        'potential_energy', 'potential_energy_signed', 'displacement',
        'total_energy', 'energy_change', 'ke_pe_ratio',
        
        # Ondas
        'dominant_freq_1', 'dominant_freq_2',
        'dominant_amplitude_1', 'dominant_amplitude_2',
        'dominant_phase_1', 'dominant_phase_2',
        'dominant_period', 'spectral_energy', 'spectral_concentration',
        'hilbert_amplitude', 'hilbert_phase', 'hilbert_freq', 'cycle_position',
        
        # Entropía/Caos
        'shannon_entropy', 'entropy_change',
        'sample_entropy', 'lyapunov',
        
        # Fractales
        'hurst', 'fractal_dim',
        
        # Mean Reversion
        'mr_zscore_20', 'mr_force_20', 'mr_k_20',
        'mr_zscore_50', 'mr_force_50', 'mr_k_50',
        'mr_zscore_100', 'mr_force_100', 'mr_k_100',
    ]


# =============================================================================
# TEST
# =============================================================================

def test_physics_features():
    """Test de features de física."""
    print("🧪 Test Features de Física...")
    
    # Datos dummy
    np.random.seed(42)
    n = 500
    
    # Simular precio con tendencia + ciclos + ruido
    t = np.linspace(0, 10, n)
    trend = 50000 + t * 1000
    cycle = 2000 * np.sin(2 * np.pi * t / 2)  # Ciclo de 2 unidades
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
    
    # Calcular features
    df_physics = calculate_physics_features(df)
    
    # Verificar
    physics_cols = get_physics_feature_columns()
    available = [c for c in physics_cols if c in df_physics.columns]
    
    print(f"✅ Features solicitadas: {len(physics_cols)}")
    print(f"✅ Features disponibles: {len(available)}")
    
    # Mostrar estadísticas
    print(f"\n📊 Estadísticas de muestra:")
    for col in available[:8]:
        val = df_physics[col].dropna()
        print(f"   {col}: mean={val.mean():.4f}, std={val.std():.4f}")
    
    # Verificar Hurst (debería ser > 0.5 por la tendencia)
    hurst_mean = df_physics['hurst'].iloc[-100:].mean()
    print(f"\n🎯 Hurst (últimos 100): {hurst_mean:.3f}")
    print(f"   {'✅ Trending (H > 0.5)' if hurst_mean > 0.5 else '⚠️ Mean reverting (H < 0.5)'}")
    
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_physics_features()
