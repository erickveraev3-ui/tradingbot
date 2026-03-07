"""
Script 09: Back to Basics - Modelo Simplificado.

Diagnóstico del 08:
- 147 features = demasiado ruido
- 27 trades = sin significancia estadística
- Sharpe -19.74 = environment mal calibrado

Solución:
- Solo 20 features CORE que sabemos funcionan
- Environment recalibrado
- Forzar más trades
- Validación estadística
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from loguru import logger
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURES ESENCIALES (Solo 20)
# =============================================================================

def calculate_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Solo las 20 features que REALMENTE aportan señal.
    Basado en investigación y práctica de trading.
    """
    df = df.copy()
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === 1. RETURNS (3) ===
    df['ret_1'] = close.pct_change(1)
    df['ret_4'] = close.pct_change(4)
    df['ret_24'] = close.pct_change(24)
    
    # === 2. VOLATILIDAD (3) ===
    # ATR normalizado
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    atr = pd.Series(tr).rolling(14).mean()
    df['atr_pct'] = atr / close
    df['atr'] = atr  # Para stops
    
    # Bollinger position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_position'] = (close - sma20) / (2 * std20 + 1e-10)
    
    # Volatility ratio
    df['vol_ratio'] = std20 / close.rolling(50).std()
    
    # === 3. TENDENCIA (4) ===
    # ADX
    plus_dm = np.where((high.diff() > 0) & (high.diff() > -low.diff()), high.diff(), 0)
    minus_dm = np.where((low.diff() < 0) & (-low.diff() > high.diff()), -low.diff(), 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = pd.Series(dx).rolling(14).mean() / 100  # Normalizado 0-1
    
    # EMA trend
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['ema_trend'] = (ema12 - ema26) / close
    
    # EMA 50/200 cross
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    df['ema_long_trend'] = (ema50 - ema200) / close
    
    # Hurst exponent (simplificado)
    def simple_hurst(series, window=100):
        result = np.zeros(len(series))
        for i in range(window, len(series)):
            seg = series[i-window:i].values
            lags = range(2, min(20, window//4))
            tau = []
            for lag in lags:
                tau.append(np.std(np.subtract(seg[lag:], seg[:-lag])))
            if len(tau) > 1 and all(t > 0 for t in tau):
                poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                result[i] = poly[0]
            else:
                result[i] = 0.5
        return result
    
    df['hurst'] = simple_hurst(close)
    
    # === 4. MOMENTUM (3) ===
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - 100 / (1 + rs)) / 100  # Normalizado 0-1
    
    # MACD histogram normalizado
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    df['macd_hist'] = (macd - macd_signal) / close
    
    # Momentum físico (precio × volumen change)
    vol_norm = volume / volume.rolling(20).mean()
    velocity = close.pct_change(5)
    df['momentum_phys'] = vol_norm * velocity
    
    # === 5. VOLUMEN (3) ===
    # OBV momentum
    obv = (np.sign(close.diff()) * volume).cumsum()
    df['obv_momentum'] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-10)
    
    # CVD approximation
    cvd = np.where(close > df['open'], volume, -volume).cumsum()
    df['cvd_momentum'] = pd.Series(cvd).diff(10) / volume.rolling(20).mean()
    
    # Volume spike
    df['vol_spike'] = volume / volume.rolling(20).mean()
    
    # === 6. ESTRUCTURA (4) ===
    # Distance to recent high/low
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    range_20 = high_20 - low_20
    df['dist_high'] = (high_20 - close) / (range_20 + 1e-10)
    df['dist_low'] = (close - low_20) / (range_20 + 1e-10)
    
    # Fibonacci 50% distance
    fib_50 = (high_20 + low_20) / 2
    df['fib_50_dist'] = (close - fib_50) / (range_20 + 1e-10)
    
    # Market structure (simplified)
    higher_high = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
    lower_low = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
    df['market_structure'] = higher_high.astype(int) - lower_low.astype(int)
    df['market_structure'] = df['market_structure'].rolling(5).mean()
    
    # Limpiar
    df = df.ffill().bfill()
    
    return df


def get_core_feature_columns() -> List[str]:
    """Lista de features core."""
    return [
        # Returns
        'ret_1', 'ret_4', 'ret_24',
        # Volatilidad
        'atr_pct', 'bb_position', 'vol_ratio',
        # Tendencia
        'adx', 'ema_trend', 'ema_long_trend', 'hurst',
        # Momentum
        'rsi', 'macd_hist', 'momentum_phys',
        # Volumen
        'obv_momentum', 'cvd_momentum', 'vol_spike',
        # Estructura
        'dist_high', 'dist_low', 'fib_50_dist', 'market_structure'
    ]


# =============================================================================
# DATASET SIMPLIFICADO
# =============================================================================

class CoreDataset(Dataset):
    """Dataset con solo 20 features core."""
    
    def __init__(self, df_4h: pd.DataFrame, seq_len: int = 50):
        self.seq_len = seq_len
        self.feature_cols = get_core_feature_columns()
        
        # Calcular features
        logger.info("📊 Calculando 20 features CORE...")
        df = calculate_core_features(df_4h)
        
        # Verificar
        available = [c for c in self.feature_cols if c in df.columns]
        logger.info(f"   Features disponibles: {len(available)}/20")
        
        # Preparar datos
        self.data = df[available].values.astype(np.float32)
        
        # Normalizar
        mean = np.nanmean(self.data, axis=0)
        std = np.nanstd(self.data, axis=0) + 1e-8
        self.data = (self.data - mean) / std
        self.data = np.clip(self.data, -3, 3)
        self.data = np.nan_to_num(self.data, 0)
        
        # Precios y ATR
        self.prices = df['close'].values
        self.atrs = df['atr'].values
        
        # Returns futuros
        self.future_ret_1 = self._future_returns(1)
        self.future_ret_3 = self._future_returns(3)
        self.future_ret_6 = self._future_returns(6)
        
        # Régimen basado en Hurst
        self.regimes = self._calc_regimes(df)
        
        logger.info(f"✅ CoreDataset: {len(self) } samples, {self.data.shape[1]} features")
    
    def _future_returns(self, horizon: int) -> np.ndarray:
        ret = np.zeros(len(self.prices))
        ret[:-horizon] = (self.prices[horizon:] - self.prices[:-horizon]) / self.prices[:-horizon]
        return ret.astype(np.float32)
    
    def _calc_regimes(self, df: pd.DataFrame) -> np.ndarray:
        hurst = df['hurst'].values if 'hurst' in df.columns else np.full(len(df), 0.5)
        adx = df['adx'].values if 'adx' in df.columns else np.full(len(df), 0.25)
        ret_24 = df['ret_24'].values if 'ret_24' in df.columns else np.zeros(len(df))
        
        regimes = np.zeros(len(df), dtype=np.int64)
        
        for i in range(len(df)):
            if hurst[i] > 0.55 and adx[i] > 0.25:
                if ret_24[i] > 0:
                    regimes[i] = 0  # Trending up
                else:
                    regimes[i] = 1  # Trending down
            elif hurst[i] < 0.45:
                regimes[i] = 2  # Mean reverting
            else:
                regimes[i] = 3  # Neutral/choppy
        
        return regimes
    
    def __len__(self):
        return len(self.data) - self.seq_len - 6
    
    def __getitem__(self, idx):
        idx = idx + self.seq_len
        
        x = self.data[idx - self.seq_len:idx]
        
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'future_ret': torch.tensor([
                self.future_ret_1[idx],
                self.future_ret_3[idx],
                self.future_ret_6[idx]
            ], dtype=torch.float32),
            'regime': torch.tensor(self.regimes[idx], dtype=torch.long),
            'price': torch.tensor(self.prices[idx], dtype=torch.float32),
            'atr': torch.tensor(self.atrs[idx], dtype=torch.float32)
        }


# =============================================================================
# MODELO SIMPLIFICADO
# =============================================================================

class CoreModel(nn.Module):
    """Modelo simple y efectivo."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True
        )
        
        # Attention simple
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output heads
        self.regime_head = nn.Linear(hidden_dim, 4)
        self.return_head = nn.Linear(hidden_dim, 3)  # 3 horizontes
        self.embedding_head = nn.Linear(hidden_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context = (attn_weights * lstm_out).sum(dim=1)  # [batch, hidden]
        
        return {
            'embedding': self.embedding_head(context),
            'regime_logits': self.regime_head(context),
            'return_preds': self.return_head(context)
        }


# =============================================================================
# ENVIRONMENT RECALIBRADO
# =============================================================================

@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    max_leverage: int = 3
    commission: float = 0.0004  # 0.04% (Binance con BNB)
    kelly_fraction: float = 0.25
    atr_stop: float = 2.5
    atr_trail: float = 1.8
    max_drawdown: float = 0.10
    min_position: float = 0.05
    max_position: float = 0.5
    
    # NUEVO: Incentivar trading
    inactivity_penalty: float = -0.0001  # Penalización por no operar
    min_trades_per_100_steps: int = 5  # Mínimo de trades esperado


class RecalibratedEnv:
    """Environment recalibrado para incentivar más trades."""
    
    def __init__(self, embeddings: np.ndarray, returns: np.ndarray,
                 regimes: np.ndarray, prices: np.ndarray, atrs: np.ndarray,
                 config: TradingConfig):
        
        self.embeddings = embeddings
        self.returns = returns
        self.regimes = regimes
        self.prices = prices
        self.atrs = atrs
        self.config = config
        
        self.state_dim = embeddings.shape[1] + 6  # embedding + position + regime + metrics
        self.action_dim = 1  # Posición continua [-1, 1]
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.capital = self.config.initial_capital
        self.entry_price = 0.0
        self.peak_capital = self.capital
        self.trades = 0
        self.wins = 0
        self.steps_since_trade = 0
        self.total_pnl = 0.0
        
        return self._get_state()
    
    def _get_state(self):
        emb = self.embeddings[self.current_step]
        
        extra = np.array([
            self.position,
            self.regimes[self.current_step] / 3.0,
            (self.capital - self.config.initial_capital) / self.config.initial_capital,
            self.peak_capital / self.config.initial_capital - 1,
            min(self.steps_since_trade / 20.0, 1.0),  # Urgencia de operar
            self.atrs[self.current_step] / self.prices[self.current_step]
        ], dtype=np.float32)
        
        return np.concatenate([emb, extra])
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1, 1)
        
        # Convertir a posición objetivo
        target_position = action * self.config.max_position
        
        # Calcular cambio de posición
        position_change = target_position - self.position
        
        # Precio actual y siguiente
        current_price = self.prices[self.current_step]
        
        # Comisiones por cambio de posición
        commission_cost = abs(position_change) * self.config.commission * self.capital
        
        # ¿Es un trade nuevo?
        is_new_trade = abs(position_change) > 0.05
        
        if is_new_trade:
            self.trades += 1
            self.steps_since_trade = 0
            if self.position != 0:
                self.entry_price = current_price
        else:
            self.steps_since_trade += 1
        
        # Actualizar posición
        old_position = self.position
        self.position = target_position
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        if self.current_step >= len(self.prices) - 1:
            return self._get_state(), 0.0, True, {'trades': self.trades}
        
        # Calcular PnL
        next_price = self.prices[self.current_step]
        price_return = (next_price - current_price) / current_price
        
        # PnL de la posición
        position_pnl = old_position * price_return * self.capital * self.config.max_leverage
        
        # Actualizar capital
        self.capital += position_pnl - commission_cost
        self.total_pnl += position_pnl - commission_cost
        
        # Actualizar peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Win tracking
        if position_pnl > 0 and is_new_trade:
            self.wins += 1
        
        # === REWARD RECALIBRADO ===
        reward = 0.0
        
        # 1. PnL normalizado (principal)
        reward += position_pnl / self.config.initial_capital * 100
        
        # 2. Penalización por inactividad
        if self.steps_since_trade > 10:
            reward += self.config.inactivity_penalty * (self.steps_since_trade - 10)
        
        # 3. Bonus por operar en tendencia correcta
        if self.regimes[self.current_step] == 0 and action > 0.1:  # Trending up + long
            reward += 0.001
        elif self.regimes[self.current_step] == 1 and action < -0.1:  # Trending down + short
            reward += 0.001
        
        # 4. Penalización por drawdown
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if drawdown > self.config.max_drawdown:
            reward -= 0.1
        
        # 5. Bonus pequeño por diversificación temporal
        if is_new_trade and self.trades > 0:
            avg_steps_per_trade = self.current_step / self.trades
            if 5 < avg_steps_per_trade < 30:  # Trading razonable
                reward += 0.0005
        
        # Check terminación
        done = False
        if drawdown > self.config.max_drawdown * 1.5:
            done = True
            reward -= 1.0
        
        info = {
            'trades': self.trades,
            'wins': self.wins,
            'capital': self.capital,
            'pnl': self.total_pnl,
            'drawdown': drawdown
        }
        
        return self._get_state(), reward, done, info


# =============================================================================
# SAC SIMPLIFICADO
# =============================================================================

class SimpleSAC:
    """SAC simplificado y estable."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, device: str = 'cuda'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # mean, log_std
        ).to(device)
        
        # Critics
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Target critics
        self.critic1_target = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.critic2_target = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = 100000
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> float:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.actor(state)
            mean = output[:, 0]
            log_std = torch.clamp(output[:, 1], -20, 2)
            std = log_std.exp()
            
            if evaluate:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                x = normal.rsample()
                action = torch.tanh(x)
        
        return action.cpu().numpy()[0]
    
    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size: int = 256):
        if len(self.buffer) < batch_size:
            return
        
        # Sample
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.FloatTensor([[t[1]] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([[t[2]] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.FloatTensor([[t[4]] for t in batch]).to(self.device)
        
        alpha = self.log_alpha.exp()
        
        # Critic update
        with torch.no_grad():
            next_output = self.actor(next_states)
            next_mean = next_output[:, 0:1]
            next_log_std = torch.clamp(next_output[:, 1:2], -20, 2)
            next_std = next_log_std.exp()
            
            normal = torch.distributions.Normal(next_mean, next_std)
            next_action = torch.tanh(normal.rsample())
            next_log_prob = normal.log_prob(next_action) - torch.log(1 - next_action.pow(2) + 1e-6)
            
            next_q1 = self.critic1_target(torch.cat([next_states, next_action], dim=1))
            next_q2 = self.critic2_target(torch.cat([next_states, next_action], dim=1))
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor update
        output = self.actor(states)
        mean = output[:, 0:1]
        log_std = torch.clamp(output[:, 1:2], -20, 2)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        action = torch.tanh(normal.rsample())
        log_prob = normal.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)
        
        q1 = self.critic1(torch.cat([states, action], dim=1))
        q2 = self.critic2(torch.cat([states, action], dim=1))
        q = torch.min(q1, q2)
        
        actor_loss = (alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, env, total_steps: int, batch_size: int = 256, 
              start_steps: int = 5000, log_interval: int = 10000):
        
        state = env.reset()
        episode_reward = 0
        episode_trades = 0
        
        for step in range(total_steps):
            # Acción
            if step < start_steps:
                action = np.random.uniform(-1, 1)
            else:
                action = self.select_action(state)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            self.store(state, action, reward, next_state, done)
            
            if step >= start_steps:
                self.train_step(batch_size)
            
            episode_reward += reward
            episode_trades = info['trades']
            
            if done:
                state = env.reset()
                episode_reward = 0
            else:
                state = next_state
            
            if (step + 1) % log_interval == 0:
                logger.info(f"Step {step+1}/{total_steps} | Trades: {episode_trades}")
    
    def evaluate(self, env, n_episodes: int = 10) -> dict:
        returns = []
        trades_list = []
        wins_list = []
        drawdowns = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.select_action(state, evaluate=True)
                state, _, done, info = env.step(action)
            
            ret = (info['capital'] - env.config.initial_capital) / env.config.initial_capital
            returns.append(ret)
            trades_list.append(info['trades'])
            wins_list.append(info['wins'])
            drawdowns.append(info['drawdown'])
        
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        
        total_trades = sum(trades_list)
        total_wins = sum(wins_list)
        
        return {
            'mean_return': mean_return,
            'sharpe': mean_return / std_return * np.sqrt(252),
            'win_rate': total_wins / max(total_trades, 1),
            'max_drawdown': max(drawdowns),
            'mean_trades': np.mean(trades_list),
            'total_trades': total_trades
        }
    
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("    🎯 SCRIPT 09: BACK TO BASICS (20 Features Core)")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # =========================================
    # CARGAR DATOS
    # =========================================
    logger.info("\n📥 Cargando datos...")
    
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    logger.info(f"   Datos 4H: {len(df_4h)} velas")
    
    # =========================================
    # CREAR DATASET
    # =========================================
    logger.info("\n📦 Creando CoreDataset (20 features)...")
    
    dataset = CoreDataset(df_4h, seq_len=50)
    
    # Split
    n = len(dataset)
    train_end = int(n * 0.50)
    val_end = int(n * 0.70)
    test_end = int(n * 0.85)
    
    train_idx = list(range(train_end))
    val_idx = list(range(train_end, val_end))
    holdout_idx = list(range(test_end, n))
    
    logger.info(f"   Train: {len(train_idx)}")
    logger.info(f"   Val: {len(val_idx)}")
    logger.info(f"   Holdout: {len(holdout_idx)}")
    
    # =========================================
    # ENTRENAR MODELO DE FEATURES
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 1: ENTRENANDO MODELO CORE")
    logger.info("=" * 60)
    
    model = CoreModel(input_dim=len(get_core_feature_columns()), hidden_dim=128, output_dim=64)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            future_ret = batch['future_ret'].to(device)
            regime = batch['regime'].to(device)
            
            out = model(x)
            
            loss_ret = nn.functional.huber_loss(out['return_preds'], future_ret)
            loss_regime = nn.functional.cross_entropy(out['regime_logits'], regime)
            loss = loss_ret + 0.2 * loss_regime
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                future_ret = batch['future_ret'].to(device)
                
                out = model(x)
                loss = nn.functional.huber_loss(out['return_preds'], future_ret)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Época {epoch+1}/100 | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if patience >= 20:
            logger.info(f"Early stopping en época {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    
    # =========================================
    # GENERAR EMBEDDINGS
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 2: GENERANDO EMBEDDINGS")
    logger.info("=" * 60)
    
    def generate_embeddings(model, dataset, device):
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        embeddings, returns, regimes, prices, atrs = [], [], [], [], []
        
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                out = model(x)
                
                embeddings.append(out['embedding'].cpu().numpy())
                returns.append(batch['future_ret'].numpy())
                regimes.append(batch['regime'].numpy())
                prices.append(batch['price'].numpy())
                atrs.append(batch['atr'].numpy())
        
        return {
            'embeddings': np.vstack(embeddings),
            'returns': np.vstack(returns),
            'regimes': np.concatenate(regimes),
            'prices': np.concatenate(prices),
            'atrs': np.concatenate(atrs)
        }
    
    train_data = generate_embeddings(model, train_dataset, device)
    val_data = generate_embeddings(model, val_dataset, device)
    holdout_data = generate_embeddings(model, torch.utils.data.Subset(dataset, holdout_idx), device)
    
    logger.info(f"   Embeddings shape: {train_data['embeddings'].shape}")
    
    # =========================================
    # OPTUNA
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 3: OPTIMIZACIÓN SAC (50 trials)")
    logger.info("=" * 60)
    
    def objective(trial):
        config = TradingConfig(
            max_leverage=trial.suggest_int('max_leverage', 2, 4),
            kelly_fraction=trial.suggest_float('kelly_fraction', 0.15, 0.35),
            atr_stop=trial.suggest_float('atr_stop', 2.0, 3.5),
            atr_trail=trial.suggest_float('atr_trail', 1.5, 2.5),
            max_drawdown=trial.suggest_float('max_drawdown', 0.08, 0.15),
            min_position=trial.suggest_float('min_position', 0.05, 0.15),
            max_position=trial.suggest_float('max_position', 0.3, 0.6),
            inactivity_penalty=trial.suggest_float('inactivity_penalty', -0.0005, -0.0001)
        )
        
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])
        lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
        
        env = RecalibratedEnv(
            val_data['embeddings'], val_data['returns'][:, 0],
            val_data['regimes'], val_data['prices'], val_data['atrs'], config
        )
        
        agent = SimpleSAC(state_dim=env.state_dim, hidden_dim=hidden_dim, lr=lr, device=device)
        
        try:
            agent.train(env, total_steps=20000, batch_size=128, start_steps=2000, log_interval=100000)
            metrics = agent.evaluate(env, n_episodes=5)
            
            # Score: Sharpe + bonus por trades
            score = metrics['sharpe']
            
            # Bonus por tener trades suficientes
            if metrics['mean_trades'] > 30:
                score += 0.5
            if metrics['mean_trades'] < 10:
                score -= 1.0
            
            # Bonus por retorno positivo
            if metrics['mean_return'] > 0.02:
                score += 1.0
            
            return score
            
        except Exception as e:
            return -10.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info(f"\n🎯 Mejores parámetros: {best_params}")
    
    # =========================================
    # ENTRENAR MODELO FINAL
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 4: ENTRENANDO MODELO FINAL")
    logger.info("=" * 60)
    
    # Combinar train + val
    combined_data = {
        'embeddings': np.vstack([train_data['embeddings'], val_data['embeddings']]),
        'returns': np.vstack([train_data['returns'], val_data['returns']]),
        'regimes': np.concatenate([train_data['regimes'], val_data['regimes']]),
        'prices': np.concatenate([train_data['prices'], val_data['prices']]),
        'atrs': np.concatenate([train_data['atrs'], val_data['atrs']])
    }
    
    final_config = TradingConfig(
        max_leverage=best_params['max_leverage'],
        kelly_fraction=best_params['kelly_fraction'],
        atr_stop=best_params['atr_stop'],
        atr_trail=best_params['atr_trail'],
        max_drawdown=best_params['max_drawdown'],
        min_position=best_params['min_position'],
        max_position=best_params['max_position'],
        inactivity_penalty=best_params['inactivity_penalty']
    )
    
    final_env = RecalibratedEnv(
        combined_data['embeddings'], combined_data['returns'][:, 0],
        combined_data['regimes'], combined_data['prices'], combined_data['atrs'],
        final_config
    )
    
    final_agent = SimpleSAC(
        state_dim=final_env.state_dim,
        hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'],
        device=device
    )
    
    final_agent.train(final_env, total_steps=150000, batch_size=256, 
                      start_steps=5000, log_interval=30000)
    
    # Guardar
    (root_dir / "artifacts/core").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root_dir / "artifacts/core/feature_model.pt")
    final_agent.save(str(root_dir / "artifacts/core/sac_agent.pt"))
    
    # =========================================
    # EVALUACIÓN HOLDOUT
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   📊 EVALUACIÓN FINAL EN HOLDOUT")
    logger.info("=" * 60)
    
    holdout_env = RecalibratedEnv(
        holdout_data['embeddings'], holdout_data['returns'][:, 0],
        holdout_data['regimes'], holdout_data['prices'], holdout_data['atrs'],
        final_config
    )
    
    metrics = final_agent.evaluate(holdout_env, n_episodes=20)
    
    # =========================================
    # RESUMEN
    # =========================================
    print("\n" + "=" * 70)
    print("              📊 RESULTADOS SCRIPT 09: BACK TO BASICS")
    print("=" * 70)
    
    print(f"\n📊 RESULTADOS EN HOLDOUT:")
    print(f"   Retorno medio: {metrics['mean_return']*100:+.2f}%")
    print(f"   Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"   Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"   Max drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"   Trades promedio: {metrics['mean_trades']:.0f}")
    print(f"   Total trades: {metrics['total_trades']}")
    
    print(f"\n🎯 Mejores hiperparámetros:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    # Evaluación
    print("\n" + "=" * 70)
    if metrics['sharpe'] > 1.0 and metrics['mean_return'] > 0.02 and metrics['mean_trades'] > 50:
        print("🎉 ¡EXCELENTE! Modelo listo para paper trading.")
    elif metrics['sharpe'] > 0.5 and metrics['mean_return'] > 0 and metrics['mean_trades'] > 30:
        print("✅ Modelo BUENO. Siguiente paso: añadir más datos.")
    elif metrics['mean_trades'] > 50 and metrics['sharpe'] > 0:
        print("🟡 Modelo opera bien pero retorno bajo. Ajustar estrategia.")
    else:
        print("⚠️ Modelo necesita más trabajo.")
    print("=" * 70)


if __name__ == "__main__":
    main()
