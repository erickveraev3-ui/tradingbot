"""
Agente SAC (Soft Actor-Critic) para Trading.
Acciones continuas: -1 (short 100%) a +1 (long 100%)
Maximiza retorno + entropía (exploración automática)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from loguru import logger
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class TradingConfig:
    """Configuración del entorno de trading."""
    initial_capital: float = 10000.0
    leverage: int = 3
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    max_position: float = 1.0
    reward_scaling: float = 1.0


class ReplayBuffer:
    """Buffer de experiencias para SAC (off-policy)."""
    
    def __init__(self, capacity: int = 100000):
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


class GaussianPolicy(nn.Module):
    """
    Red de política que produce una distribución gaussiana.
    Output: media y log_std de la acción continua.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Última capa con pesos pequeños
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna media y log_std."""
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea una acción usando reparametrización.
        
        Returns:
            action: Acción en [-1, 1]
            log_prob: Log probabilidad de la acción
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparametrización
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Muestreo con gradientes
        
        # Squash a [-1, 1] con tanh
        action = torch.tanh(x_t)
        
        # Calcular log_prob con corrección de tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Obtiene acción para inferencia."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        return action


class QNetwork(nn.Module):
    """Red Q (crítico) - estima Q(s, a)."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 por la acción
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 (twin para reducir overestimation)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class TradingEnvironment:
    """
    Entorno de trading para SAC.
    Acción continua: -1 (short 100%) a +1 (long 100%)
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        predictions: np.ndarray,
        prices: np.ndarray,
        config: TradingConfig = None
    ):
        self.embeddings = embeddings
        self.predictions = predictions
        self.prices = prices
        self.config = config or TradingConfig()
        
        # Calcular retornos
        self.returns = np.diff(prices) / prices[:-1]
        self.returns = np.append(self.returns, 0)
        
        self.n_steps = len(embeddings)
        
        # Estadísticas de retornos para normalizar rewards
        self.return_std = np.std(self.returns) + 1e-8
        
        self.reset()
        
        logger.info(f"🎮 TradingEnvironment creado: {self.n_steps} steps")
    
    @property
    def state_dim(self) -> int:
        """Dimensión del estado."""
        return self.embeddings.shape[1] + 3 + 5
    
    def reset(self, start_idx: int = None) -> np.ndarray:
        """
        Reinicia el entorno.
        start_idx: Índice de inicio aleatorio para variabilidad
        """
        # Inicio aleatorio para más variabilidad en el entrenamiento
        if start_idx is None:
            max_start = max(0, self.n_steps - 500)  # Mínimo 500 steps por episodio
            self.start_step = np.random.randint(0, max(1, max_start))
        else:
            self.start_step = start_idx
        
        self.current_step = self.start_step
        self.position = 0.0
        self.capital = self.config.initial_capital
        self.peak_capital = self.capital
        self.total_trades = 0
        self.total_pnl = 0.0
        self.last_action = 0.0
        self.cumulative_return = 0.0
        self.returns_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Construye el estado."""
        if self.current_step >= self.n_steps:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Embedding
        embedding = self.embeddings[self.current_step]
        
        # Predicciones (q10, q50, q90)
        preds = self.predictions[self.current_step]
        
        # Estado de cuenta normalizado
        account = np.array([
            self.position,
            np.clip(self.capital / self.config.initial_capital - 1, -1, 1),  # Retorno normalizado
            np.clip((self.capital - self.peak_capital) / self.peak_capital, -1, 0),  # Drawdown
            (self.current_step - self.start_step) / 500,  # Progreso normalizado
            self.last_action
        ], dtype=np.float32)
        
        return np.concatenate([embedding, preds, account]).astype(np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta acción continua.
        action: float en [-1, 1]
        """
        action = float(np.clip(action, -1.0, 1.0))
        
        # Discretizar ligeramente para evitar micro-cambios
        if abs(action) < 0.1:
            action = 0.0
        
        pnl = 0.0
        fee = 0.0
        
        if self.current_step < self.n_steps - 1:
            market_return = self.returns[self.current_step]
            
            # PnL basado en posición actual (antes de cambiar)
            if abs(self.position) > 0.05:
                effective_position = self.position * self.config.leverage
                pnl = self.capital * effective_position * market_return
            
            # Fee solo si cambio significativo de posición (>10%)
            position_change = abs(action - self.position)
            if position_change > 0.1:
                fee = self.capital * abs(position_change) * self.config.taker_fee
                self.total_trades += 1
        
        # Actualizar capital
        net_pnl = pnl - fee
        self.capital += net_pnl
        self.total_pnl += net_pnl
        
        # Guardar retorno para Sharpe
        step_return = net_pnl / self.config.initial_capital
        self.returns_history.append(step_return)
        self.cumulative_return += step_return
        
        # Actualizar peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Actualizar posición
        self.last_action = action
        self.position = action
        
        # Avanzar
        self.current_step += 1
        
        # Condiciones de terminación
        steps_done = self.current_step - self.start_step
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        
        done = (
            self.current_step >= self.n_steps - 1 or
            steps_done >= 500 or
            drawdown >= 0.15 or  # Stop si pierde 15%
            self.capital <= self.config.initial_capital * 0.85  # Stop loss 15%
        )
        
        # Reward
        reward = self._calculate_reward(pnl, fee, market_return)
        
        # Penalización fuerte por drawdown excesivo
        if drawdown > 0.10:
            reward -= 1.0 * drawdown  # Penalizar proporcionalmente
        
        info = {
            'capital': self.capital,
            'pnl': net_pnl,
            'position': self.position,
            'drawdown': drawdown,
            'total_trades': self.total_trades,
            'total_return': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'step': self.current_step
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, pnl: float, fee: float, market_return: float) -> float:
        """
        Reward mejorado para trading.
        Objetivos:
        1. Ganar dinero (retorno positivo)
        2. Estar posicionado en la dirección correcta
        3. No hacer overtrading
        """
        reward = 0.0
        
        # 1. Retorno normalizado (componente principal)
        net_return = (pnl - fee) / self.config.initial_capital
        reward += net_return / self.return_std * 10  # Normalizado y escalado
        
        # 2. Bonus por dirección correcta
        if abs(self.position) > 0.1 and abs(market_return) > 0.001:
            direction_correct = np.sign(self.position) == np.sign(market_return)
            if direction_correct:
                reward += 0.1  # Bonus por acertar dirección
            else:
                reward -= 0.05  # Penalización menor por fallar
        
        # 3. Penalización por cambios frecuentes de posición
        if self.total_trades > 0:
            steps_done = max(1, self.current_step - self.start_step)
            trade_frequency = self.total_trades / steps_done
            if trade_frequency > 0.1:  # Más de 1 trade cada 10 velas
                reward -= 0.02 * trade_frequency
        
        # 4. Pequeño costo por mantener posición (evita posiciones perpetuas)
        reward -= 0.001 * abs(self.position)
        
        return float(np.clip(reward, -10, 10))


class SACAgent:
    """
    Agente SAC completo.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        
        # Redes
        self.policy = GaussianPolicy(state_dim, hidden_dim).to(self.device)
        self.q_network = QNetwork(state_dim, hidden_dim).to(self.device)
        self.q_target = QNetwork(state_dim, hidden_dim).to(self.device)
        
        # Copiar pesos a target
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Optimizadores
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Entropía automática
        if auto_entropy:
            self.target_entropy = -1.0  # -dim(action)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        logger.info(f"🤖 SACAgent inicializado en {self.device}")
        logger.info(f"   State dim: {state_dim}")
        logger.info(f"   Hidden dim: {hidden_dim}")
        logger.info(f"   Auto entropy: {auto_entropy}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Selecciona acción dado un estado."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy().flatten()[0]
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Actualiza las redes con un batch del replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Samplear batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # ===== Actualizar Q networks =====
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next, q2_next = self.q_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.q_network(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ===== Actualizar Policy =====
        new_actions, log_probs = self.policy.sample(states)
        q1_new, q2_new = self.q_network(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ===== Actualizar Alpha (entropía automática) =====
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ===== Soft update de target network =====
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if self.auto_entropy else 0.0
        }
    
    def train(
        self,
        env: TradingEnvironment,
        total_steps: int = 100000,
        batch_size: int = 256,
        start_steps: int = 1000,
        update_every: int = 1,
        log_interval: int = 1000,
        eval_interval: int = 5000
    ) -> Dict[str, List]:
        """
        Entrena el agente SAC.
        """
        logger.info(f"🚀 Iniciando entrenamiento SAC: {total_steps} steps")
        
        history = {
            'rewards': [],
            'q_loss': [],
            'policy_loss': [],
            'alpha': [],
            'eval_returns': []
        }
        
        state = env.reset()
        episode_reward = 0
        episode_rewards = []
        
        for step in range(total_steps):
            # Seleccionar acción
            if step < start_steps:
                action = np.random.uniform(-1, 1)  # Exploración inicial
            else:
                action = self.select_action(state)
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            
            # Guardar en buffer
            self.replay_buffer.push(state, [action], reward, next_state, float(done))
            
            episode_reward += reward
            state = next_state
            
            if done:
                episode_rewards.append(episode_reward)
                history['rewards'].append(episode_reward)
                episode_reward = 0
                state = env.reset()
            
            # Actualizar redes
            if step >= start_steps and step % update_every == 0:
                losses = self.update(batch_size)
                if losses:
                    history['q_loss'].append(losses['q_loss'])
                    history['policy_loss'].append(losses['policy_loss'])
                    history['alpha'].append(losses['alpha'])
            
            # Logging
            if (step + 1) % log_interval == 0 and episode_rewards:
                mean_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Step {step + 1}/{total_steps} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Alpha: {self.alpha:.4f} | "
                    f"Buffer: {len(self.replay_buffer)}"
                )
            
            # Evaluación
            if (step + 1) % eval_interval == 0:
                eval_return = self.evaluate(env, n_episodes=3)
                history['eval_returns'].append(eval_return)
                logger.info(f"📊 Eval Return: {eval_return:.2%}")
        
        logger.info("✅ Entrenamiento SAC completado")
        
        return history
    
    def evaluate(self, env: TradingEnvironment, n_episodes: int = 5) -> float:
        """Evalúa el agente sin exploración."""
        returns = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.select_action(state, deterministic=True)
                state, _, done, info = env.step(action)
            
            returns.append(info['total_return'])
        
        return np.mean(returns)
    
    def save(self, path: str):
        """Guarda el agente."""
        torch.save({
            'policy_state': self.policy.state_dict(),
            'q_network_state': self.q_network.state_dict(),
            'q_target_state': self.q_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None,
            'alpha': self.alpha
        }, path)
        logger.info(f"💾 SAC guardado en: {path}")
    
    def load(self, path: str):
        """Carga el agente."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.q_target.load_state_dict(checkpoint['q_target_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        if checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']
        logger.info(f"📂 SAC cargado desde: {path}")


def test_sac_agent():
    """Prueba básica del agente SAC."""
    
    # Datos dummy
    n_samples = 1000
    embedding_dim = 128
    
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    predictions = np.random.randn(n_samples, 3).astype(np.float32) * 0.02
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    
    # Crear entorno
    env = TradingEnvironment(embeddings, predictions, prices)
    
    print(f"🎮 Entorno creado:")
    print(f"   State dim: {env.state_dim}")
    print(f"   Steps: {env.n_steps}")
    
    # Crear agente
    agent = SACAgent(state_dim=env.state_dim, hidden_dim=128)
    
    # Test rápido
    state = env.reset()
    print(f"\n🧪 Test básico:")
    print(f"   State shape: {state.shape}")
    
    action = agent.select_action(state)
    print(f"   Action: {action:.4f}")
    
    next_state, reward, done, info = env.step(action)
    print(f"   Reward: {reward:.4f}")
    print(f"   Capital: ${info['capital']:.2f}")
    
    # Entrenar un poco
    print(f"\n🏋️ Mini entrenamiento (500 steps)...")
    history = agent.train(env, total_steps=500, log_interval=100, start_steps=100)
    
    print(f"\n✅ Test completado")
    print(f"   Episodes: {len(history['rewards'])}")
    if history['rewards']:
        print(f"   Last reward: {history['rewards'][-1]:.4f}")


if __name__ == "__main__":
    test_sac_agent()
