"""
SAC Avanzado con:
- Kelly Criterion para position sizing
- Trailing Stop dinámico basado en ATR
- Adaptación al régimen de mercado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from loguru import logger
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class AdvancedTradingConfig:
    """Configuración avanzada de trading."""
    initial_capital: float = 10000.0
    max_leverage: int = 3
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    
    # Kelly Criterion
    kelly_fraction: float = 0.5  # Fracción de Kelly (más conservador)
    min_position: float = 0.1   # Mínimo 10% si hay señal
    max_position: float = 1.0   # Máximo 100%
    
    # Trailing Stop
    atr_multiplier_stop: float = 2.0   # Stop loss a 2x ATR
    atr_multiplier_trail: float = 1.5  # Trailing a 1.5x ATR
    
    # Risk Management
    max_drawdown: float = 0.15  # Máximo 15% drawdown
    daily_loss_limit: float = 0.05  # Máximo 5% pérdida diaria


class ReplayBuffer:
    """Buffer de experiencias."""
    
    def __init__(self, capacity: int = 200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class KellyCriterion:
    """Calcula el tamaño óptimo de posición usando Kelly Criterion."""
    
    def __init__(self, lookback: int = 100, kelly_fraction: float = 0.5):
        self.lookback = lookback
        self.kelly_fraction = kelly_fraction
        self.trade_history = deque(maxlen=lookback)
    
    def add_trade(self, pnl_pct: float):
        """Añade resultado de trade (como %)."""
        self.trade_history.append(pnl_pct)
    
    def calculate(self, pred_q10: float, pred_q50: float, pred_q90: float) -> float:
        """
        Calcula tamaño de posición óptimo.
        
        Args:
            pred_q10, pred_q50, pred_q90: Predicciones de retorno
            
        Returns:
            Fracción de capital a usar (0 a 1)
        """
        if len(self.trade_history) < 10:
            # Sin historial suficiente, usar predicciones directamente
            # Calcular probabilidad implícita de ganar
            if pred_q50 > 0:
                p_win = 0.5 + (pred_q90 - pred_q50) / (pred_q90 - pred_q10 + 1e-8) * 0.3
            else:
                p_win = 0.5 - (pred_q50 - pred_q10) / (pred_q90 - pred_q10 + 1e-8) * 0.3
            
            # Ratio ganancia/pérdida esperado
            if pred_q50 > 0:
                b = abs(pred_q90) / (abs(pred_q10) + 1e-8)
            else:
                b = abs(pred_q10) / (abs(pred_q90) + 1e-8)
        else:
            # Usar historial de trades
            trades = np.array(self.trade_history)
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 0.1  # Mínimo si no hay suficientes datos
            
            p_win = len(wins) / len(trades)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            b = avg_win / (avg_loss + 1e-8)
        
        # Kelly Formula: f* = (p * b - q) / b
        p_win = np.clip(p_win, 0.1, 0.9)
        b = np.clip(b, 0.5, 3.0)
        q = 1 - p_win
        
        kelly = (p_win * b - q) / b
        
        # Aplicar fracción de Kelly (más conservador)
        kelly *= self.kelly_fraction
        
        # Ajustar por confianza en la predicción (spread del intervalo)
        confidence = 1 - (pred_q90 - pred_q10) / 0.1  # Normalizado
        confidence = np.clip(confidence, 0.3, 1.0)
        kelly *= confidence
        
        return np.clip(kelly, 0.0, 1.0)


class TrailingStopManager:
    """Gestiona trailing stops dinámicos basados en ATR."""
    
    def __init__(
        self,
        atr_multiplier_stop: float = 2.0,
        atr_multiplier_trail: float = 1.5
    ):
        self.atr_mult_stop = atr_multiplier_stop
        self.atr_mult_trail = atr_multiplier_trail
        self.reset()
    
    def reset(self):
        self.entry_price = None
        self.position = 0
        self.highest_price = None
        self.lowest_price = None
        self.stop_price = None
    
    def open_position(self, price: float, position: float, atr: float):
        """Abre una nueva posición."""
        self.entry_price = price
        self.position = position
        self.highest_price = price
        self.lowest_price = price
        
        # Stop inicial basado en ATR
        if position > 0:  # Long
            self.stop_price = price - atr * self.atr_mult_stop
        else:  # Short
            self.stop_price = price + atr * self.atr_mult_stop
    
    def update(self, current_price: float, atr: float) -> Tuple[bool, Optional[float]]:
        """
        Actualiza el trailing stop.
        
        Returns:
            (should_close, stop_price)
        """
        if self.entry_price is None:
            return False, None
        
        # Actualizar precio extremo
        if self.position > 0:  # Long
            if current_price > self.highest_price:
                self.highest_price = current_price
                # Mover stop hacia arriba
                new_stop = current_price - atr * self.atr_mult_trail
                self.stop_price = max(self.stop_price, new_stop)
            
            # Verificar si se activó el stop
            if current_price <= self.stop_price:
                return True, self.stop_price
                
        else:  # Short
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                # Mover stop hacia abajo
                new_stop = current_price + atr * self.atr_mult_trail
                self.stop_price = min(self.stop_price, new_stop)
            
            # Verificar si se activó el stop
            if current_price >= self.stop_price:
                return True, self.stop_price
        
        return False, self.stop_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calcula PnL no realizado."""
        if self.entry_price is None:
            return 0.0
        
        if self.position > 0:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


class GaussianPolicy(nn.Module):
    """Red de política gaussiana mejorada."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_regimes: int = 4
    ):
        super().__init__()
        
        # Red compartida
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Embedding de régimen
        self.regime_embedding = nn.Embedding(n_regimes, 32)
        
        # Cabeza de política condicionada al régimen
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # mean, log_std
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, regime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
            regime: (batch,) - índice del régimen
        """
        shared = self.shared(state)
        regime_emb = self.regime_embedding(regime)
        
        combined = torch.cat([shared, regime_emb], dim=-1)
        output = self.policy_head(combined)
        
        mean = output[:, 0:1]
        log_std = torch.clamp(output[:, 1:2], -20, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, regime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, regime)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, regime: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(state, regime)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        return torch.tanh(x)


class QNetwork(nn.Module):
    """Twin Q-Network."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class AdvancedTradingEnv:
    """
    Entorno de trading avanzado con:
    - Multi-timeframe input
    - Kelly position sizing
    - Trailing stops
    - Régimen de mercado
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        predictions_4h: np.ndarray,
        predictions_12h: np.ndarray,
        predictions_24h: np.ndarray,
        regimes: np.ndarray,
        prices: np.ndarray,
        atrs: np.ndarray,
        config: AdvancedTradingConfig = None
    ):
        self.embeddings = embeddings
        self.predictions_4h = predictions_4h
        self.predictions_12h = predictions_12h
        self.predictions_24h = predictions_24h
        self.regimes = regimes
        self.prices = prices
        self.atrs = atrs
        self.config = config or AdvancedTradingConfig()
        
        # Retornos
        self.returns = np.diff(prices) / prices[:-1]
        self.returns = np.append(self.returns, 0)
        
        self.n_steps = len(embeddings)
        
        # Componentes
        self.kelly = KellyCriterion(kelly_fraction=self.config.kelly_fraction)
        self.trailing_stop = TrailingStopManager(
            self.config.atr_multiplier_stop,
            self.config.atr_multiplier_trail
        )
        
        self.reset()
        
        logger.info(f"🎮 AdvancedTradingEnv: {self.n_steps} steps")
    
    @property
    def state_dim(self) -> int:
        # embedding + preds(9) + regime_onehot(4) + account(6)
        return self.embeddings.shape[1] + 9 + 4 + 6
    
    def reset(self, start_idx: int = None) -> Tuple[np.ndarray, int]:
        # Random start
        if start_idx is None:
            max_start = max(0, self.n_steps - 600)
            self.start_step = np.random.randint(0, max(1, max_start))
        else:
            self.start_step = start_idx
        
        self.current_step = self.start_step
        self.position = 0.0
        self.position_size = 0.0
        self.capital = self.config.initial_capital
        self.peak_capital = self.capital
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.steps_in_day = 0
        
        self.kelly.trade_history.clear()
        self.trailing_stop.reset()
        
        state = self._get_state()
        regime = self.regimes[self.current_step]
        
        return state, regime
    
    def _get_state(self) -> np.ndarray:
        if self.current_step >= self.n_steps:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Embedding
        emb = self.embeddings[self.current_step]
        
        # Predicciones (todas las temporalidades)
        pred_4h = self.predictions_4h[self.current_step]
        pred_12h = self.predictions_12h[self.current_step]
        pred_24h = self.predictions_24h[self.current_step]
        preds = np.concatenate([pred_4h, pred_12h, pred_24h])
        
        # Régimen one-hot
        regime_onehot = np.zeros(4)
        regime_onehot[self.regimes[self.current_step]] = 1
        
        # Estado de cuenta
        account = np.array([
            self.position,
            self.position_size,
            self.capital / self.config.initial_capital - 1,
            (self.capital - self.peak_capital) / self.peak_capital,
            self.daily_pnl / self.config.initial_capital,
            self.trailing_stop.get_unrealized_pnl(self.prices[self.current_step]) if self.position != 0 else 0
        ], dtype=np.float32)
        
        return np.concatenate([emb, preds, regime_onehot, account]).astype(np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, int, dict]:
        """
        Args:
            action: -1 (full short) a +1 (full long), indica DIRECCIÓN
        
        Returns:
            next_state, reward, done, next_regime, info
        """
        action = float(np.clip(action, -1.0, 1.0))
        
        # Obtener predicciones actuales para Kelly
        pred = self.predictions_12h[self.current_step]
        
        # Calcular tamaño óptimo con Kelly
        kelly_size = self.kelly.calculate(pred[0], pred[1], pred[2])
        
        # Dirección de la acción determina long/short
        desired_direction = np.sign(action) if abs(action) > 0.1 else 0
        
        # Tamaño final = Kelly * intensidad de la señal
        signal_intensity = abs(action)
        desired_size = kelly_size * signal_intensity * self.config.max_leverage
        
        # Aplicar límites
        desired_size = np.clip(desired_size, 0, self.config.max_position * self.config.max_leverage)
        
        desired_position = desired_direction * desired_size
        
        # Variables de reward
        pnl = 0.0
        fee = 0.0
        stop_triggered = False
        
        current_price = self.prices[self.current_step]
        current_atr = self.atrs[self.current_step]
        market_return = self.returns[self.current_step]
        
        # Verificar trailing stop
        if self.position != 0:
            stop_triggered, _ = self.trailing_stop.update(current_price, current_atr)
            
            if stop_triggered:
                # Cerrar posición por stop
                unrealized = self.trailing_stop.get_unrealized_pnl(current_price)
                pnl = self.capital * abs(self.position_size) * unrealized
                fee = self.capital * abs(self.position_size) * self.config.taker_fee
                
                # Registrar trade
                self.kelly.add_trade(unrealized)
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                
                self.position = 0
                self.position_size = 0
                self.trailing_stop.reset()
                desired_position = 0  # No abrir nueva posición inmediatamente después del stop
        
        # PnL de posición existente (si no se cerró por stop)
        if self.position != 0 and not stop_triggered:
            pnl = self.capital * self.position_size * self.position * market_return
        
        # Cambio de posición
        if abs(desired_position - self.position) > 0.05 and not stop_triggered:
            # Cerrar posición anterior
            if self.position != 0:
                fee += self.capital * abs(self.position_size) * self.config.taker_fee
                # Registrar trade
                unrealized = self.trailing_stop.get_unrealized_pnl(current_price)
                self.kelly.add_trade(unrealized)
                self.total_trades += 1
                if unrealized > 0:
                    self.winning_trades += 1
                self.trailing_stop.reset()
            
            # Abrir nueva posición
            if desired_position != 0:
                fee += self.capital * abs(desired_size) * self.config.taker_fee
                self.trailing_stop.open_position(
                    current_price,
                    np.sign(desired_position),
                    current_atr
                )
            
            self.position = np.sign(desired_position)
            self.position_size = abs(desired_size)
        
        # Actualizar capital
        net_pnl = pnl - fee
        self.capital += net_pnl
        self.total_pnl += net_pnl
        self.daily_pnl += net_pnl
        
        # Reset diario (cada 6 velas de 4H = 24 horas)
        self.steps_in_day += 1
        if self.steps_in_day >= 6:
            self.daily_pnl = 0
            self.steps_in_day = 0
        
        # Actualizar peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Avanzar
        self.current_step += 1
        
        # Condiciones de término
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        steps_done = self.current_step - self.start_step
        
        done = (
            self.current_step >= self.n_steps - 1 or
            steps_done >= 500 or
            drawdown >= self.config.max_drawdown or
            self.daily_pnl <= -self.config.initial_capital * self.config.daily_loss_limit
        )
        
        # Reward
        reward = self._calculate_reward(net_pnl, market_return, drawdown, stop_triggered)
        
        # Info
        info = {
            'capital': self.capital,
            'pnl': net_pnl,
            'position': self.position,
            'position_size': self.position_size,
            'kelly_size': kelly_size,
            'drawdown': drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_return': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'stop_triggered': stop_triggered
        }
        
        next_state = self._get_state()
        next_regime = self.regimes[min(self.current_step, self.n_steps - 1)]
        
        return next_state, reward, done, next_regime, info
    
    def _calculate_reward(
        self,
        net_pnl: float,
        market_return: float,
        drawdown: float,
        stop_triggered: bool
    ) -> float:
        """Reward function optimizada para Sharpe ratio."""
        
        # Retorno normalizado
        ret = net_pnl / self.config.initial_capital
        reward = ret * 100  # Escalar
        
        # Bonus por dirección correcta
        if self.position != 0 and abs(market_return) > 0.002:
            if np.sign(self.position) == np.sign(market_return):
                reward += 0.2
            else:
                reward -= 0.1
        
        # Penalización por drawdown
        if drawdown > 0.05:
            reward -= drawdown * 2
        
        # Bonus por usar stop correctamente
        if stop_triggered and net_pnl < 0:
            reward += 0.1  # Bien por cortar pérdidas
        
        # Penalización por no usar Kelly correctamente
        if self.position_size > 0.8 * self.config.max_leverage:
            reward -= 0.1  # Penalizar over-leveraging
        
        return float(np.clip(reward, -10, 10))


class SACAdvanced:
    """SAC con soporte para régimen de mercado."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_regimes: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.n_regimes = n_regimes
        
        # Networks
        self.policy = GaussianPolicy(state_dim, hidden_dim, n_regimes).to(self.device)
        self.q_network = QNetwork(state_dim, hidden_dim).to(self.device)
        self.q_target = QNetwork(state_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Auto entropy
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha = 0.2
        
        # Replay buffer (ahora guarda régimen también)
        self.replay_buffer = deque(maxlen=200000)
        
        logger.info(f"🤖 SACAdvanced inicializado en {self.device}")
    
    def push_experience(self, state, action, reward, next_state, done, regime, next_regime):
        self.replay_buffer.append((state, action, reward, next_state, done, regime, next_regime))
    
    def select_action(self, state: np.ndarray, regime: int, deterministic: bool = False) -> float:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        regime_t = torch.LongTensor([regime]).to(self.device)
        
        with torch.no_grad():
            action = self.policy.get_action(state_t, regime_t, deterministic)
        
        return action.cpu().numpy().flatten()[0]
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones, regimes, next_regimes = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        regimes = torch.LongTensor(np.array(regimes)).to(self.device)
        next_regimes = torch.LongTensor(np.array(next_regimes)).to(self.device)
        
        # Update Q
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states, next_regimes)
            q1_next, q2_next = self.q_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.q_network(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update Policy
        new_actions, log_probs = self.policy.sample(states, regimes)
        q1_new, q2_new = self.q_network(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # Soft update
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha
        }
    
    def train(
        self,
        env: AdvancedTradingEnv,
        total_steps: int = 300000,
        batch_size: int = 256,
        start_steps: int = 10000,
        log_interval: int = 10000,
        eval_interval: int = 25000
    ) -> Dict[str, List]:
        logger.info(f"🚀 Entrenamiento SAC Advanced: {total_steps} steps")
        
        history = {'rewards': [], 'q_loss': [], 'returns': [], 'win_rates': []}
        
        state, regime = env.reset()
        episode_reward = 0
        episode_rewards = []
        
        for step in range(total_steps):
            if step < start_steps:
                action = np.random.uniform(-1, 1)
            else:
                action = self.select_action(state, regime)
            
            next_state, reward, done, next_regime, info = env.step(action)
            
            self.push_experience(state, action, reward, next_state, float(done), regime, next_regime)
            
            episode_reward += reward
            state = next_state
            regime = next_regime
            
            if done:
                episode_rewards.append(episode_reward)
                history['rewards'].append(episode_reward)
                history['returns'].append(info['total_return'])
                history['win_rates'].append(info['win_rate'])
                episode_reward = 0
                state, regime = env.reset()
            
            if step >= start_steps:
                losses = self.update(batch_size)
                if losses:
                    history['q_loss'].append(losses['q_loss'])
            
            if (step + 1) % log_interval == 0 and episode_rewards:
                recent_returns = history['returns'][-10:] if history['returns'] else [0]
                recent_wr = history['win_rates'][-10:] if history['win_rates'] else [0]
                logger.info(
                    f"Step {step + 1}/{total_steps} | "
                    f"Return: {np.mean(recent_returns)*100:+.1f}% | "
                    f"WinRate: {np.mean(recent_wr)*100:.0f}% | "
                    f"Alpha: {self.alpha:.3f}"
                )
        
        logger.info("✅ Entrenamiento completado")
        return history
    
    def evaluate(self, env: AdvancedTradingEnv, n_episodes: int = 10) -> Dict[str, float]:
        returns = []
        win_rates = []
        drawdowns = []
        trades = []
        
        for _ in range(n_episodes):
            state, regime = env.reset()
            done = False
            
            while not done:
                action = self.select_action(state, regime, deterministic=True)
                state, _, done, regime, info = env.step(action)
            
            returns.append(info['total_return'])
            win_rates.append(info['win_rate'])
            drawdowns.append(info['drawdown'])
            trades.append(info['total_trades'])
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_win_rate': np.mean(win_rates),
            'max_drawdown': np.max(drawdowns),
            'mean_trades': np.mean(trades),
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        }
    
    def save(self, path: str):
        torch.save({
            'policy_state': self.policy.state_dict(),
            'q_network_state': self.q_network.state_dict(),
            'q_target_state': self.q_target.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha
        }, path)
        logger.info(f"💾 Modelo guardado: {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.q_target.load_state_dict(checkpoint['q_target_state'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']
        logger.info(f"📂 Modelo cargado: {path}")


def test_advanced():
    """Test del sistema avanzado."""
    print("🧪 Test SAC Advanced...")
    
    n = 1000
    emb_dim = 160
    
    embeddings = np.random.randn(n, emb_dim).astype(np.float32)
    pred_4h = np.random.randn(n, 3).astype(np.float32) * 0.02
    pred_12h = np.random.randn(n, 3).astype(np.float32) * 0.03
    pred_24h = np.random.randn(n, 3).astype(np.float32) * 0.04
    regimes = np.random.randint(0, 4, n)
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    atrs = np.abs(np.random.randn(n) * 500 + 1000)
    
    env = AdvancedTradingEnv(
        embeddings, pred_4h, pred_12h, pred_24h, regimes, prices, atrs
    )
    
    print(f"✅ State dim: {env.state_dim}")
    
    agent = SACAdvanced(state_dim=env.state_dim, hidden_dim=128)
    
    # Mini training
    print("🏋️ Mini training (1000 steps)...")
    history = agent.train(env, total_steps=1000, log_interval=500, start_steps=200)
    
    print(f"✅ Test completado")


if __name__ == "__main__":
    test_advanced()
