"""
Script 04: Backtest Profesional con VectorBT.
Evalúa el sistema completo con métricas institucionales.

Uso:
    python scripts/04_backtest.py
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger
from datetime import datetime

from src.features.indicators import calculate_all_indicators
from src.models.multi_timeframe_detector import MultiTimeframeDetector, MultiTimeframeDataset
from src.models.sac_advanced import SACAdvanced, AdvancedTradingEnv, AdvancedTradingConfig


def load_models(device):
    """Carga los modelos entrenados."""
    logger.info("📂 Cargando modelos...")
    
    # Cargar detector
    detector_path = root_dir / "artifacts/multi_timeframe/detector.pt"
    detector_checkpoint = torch.load(detector_path, map_location=device)
    
    detector = MultiTimeframeDetector(
        input_dim_1d=14,
        input_dim_4h=14,
        input_dim_1h=14
    )
    detector.load_state_dict(detector_checkpoint['model_state'])
    detector.to(device)
    detector.eval()
    
    # Cargar SAC
    sac_path = root_dir / "artifacts/sac_advanced/agent.pt"
    
    # Necesitamos saber el state_dim
    state_dim = 160 + 9 + 4 + 6  # embedding + preds + regime + account
    
    agent = SACAdvanced(state_dim=state_dim, hidden_dim=256, device=device)
    agent.load(str(sac_path))
    
    logger.info("✅ Modelos cargados")
    
    return detector, agent


def run_backtest_simulation(agent, env, deterministic=True):
    """Ejecuta simulación completa del backtest."""
    
    # Reset al inicio (no aleatorio)
    state, regime = env.reset(start_idx=0)
    
    # Historiales
    positions = []
    capitals = []
    returns = []
    timestamps = []
    actions = []
    regimes_hist = []
    
    step = 0
    done = False
    
    while not done and step < env.n_steps - 1:
        # Obtener acción
        action = agent.select_action(state, regime, deterministic=deterministic)
        
        # Ejecutar
        next_state, reward, done, next_regime, info = env.step(action)
        
        # Guardar
        positions.append(info['position'] * info['position_size'])
        capitals.append(info['capital'])
        returns.append(info['pnl'] / env.config.initial_capital)
        actions.append(action)
        regimes_hist.append(regime)
        
        state = next_state
        regime = next_regime
        step += 1
        
        # Forzar continuar aunque done (para backtest completo)
        if done and step < env.n_steps - 100:
            done = False
    
    return {
        'positions': np.array(positions),
        'capitals': np.array(capitals),
        'returns': np.array(returns),
        'actions': np.array(actions),
        'regimes': np.array(regimes_hist),
        'final_info': info
    }


def calculate_metrics(results, prices, initial_capital=10000):
    """Calcula métricas profesionales."""
    
    returns = results['returns']
    capitals = results['capitals']
    positions = results['positions']
    
    # Métricas básicas
    total_return = (capitals[-1] - initial_capital) / initial_capital
    
    # Sharpe Ratio (anualizado, asumiendo 4H = 6 períodos/día)
    periods_per_year = 6 * 365
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    
    # Sortino Ratio (solo penaliza downside)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) + 1e-8
    sortino = mean_return / downside_std * np.sqrt(periods_per_year)
    
    # Calmar Ratio
    running_max = np.maximum.accumulate(capitals)
    drawdowns = (running_max - capitals) / running_max
    max_drawdown = np.max(drawdowns)
    calmar = (total_return * (periods_per_year / len(returns))) / (max_drawdown + 1e-8)
    
    # Win Rate
    winning_periods = np.sum(returns > 0)
    total_periods = np.sum(returns != 0)
    win_rate = winning_periods / (total_periods + 1e-8)
    
    # Profit Factor
    gross_profits = np.sum(returns[returns > 0])
    gross_losses = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profits / (gross_losses + 1e-8)
    
    # Número de trades (cambios de posición significativos)
    position_changes = np.abs(np.diff(positions))
    n_trades = np.sum(position_changes > 0.1)
    
    # Average Trade
    avg_trade = total_return / (n_trades + 1e-8)
    
    # Exposure (% del tiempo con posición)
    exposure = np.mean(np.abs(positions) > 0.1)
    
    # Best/Worst periods
    best_period = np.max(returns) * 100
    worst_period = np.min(returns) * 100
    
    # Consecutive wins/losses
    signs = np.sign(returns)
    max_consec_wins = max_consecutive(signs, 1)
    max_consec_losses = max_consecutive(signs, -1)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': n_trades,
        'avg_trade': avg_trade,
        'exposure': exposure,
        'best_period': best_period,
        'worst_period': worst_period,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'final_capital': capitals[-1]
    }


def max_consecutive(arr, value):
    """Cuenta máximo de valores consecutivos."""
    max_count = 0
    current_count = 0
    for x in arr:
        if x == value:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def compare_with_buyhold(capitals, prices, initial_capital):
    """Compara con estrategia Buy & Hold."""
    
    # Normalizar precios al capital inicial
    buyhold_capitals = initial_capital * prices / prices[0]
    
    # Ajustar longitud
    min_len = min(len(capitals), len(buyhold_capitals))
    capitals = capitals[:min_len]
    buyhold_capitals = buyhold_capitals[:min_len]
    
    # Retornos
    strategy_return = (capitals[-1] - initial_capital) / initial_capital
    buyhold_return = (buyhold_capitals[-1] - initial_capital) / initial_capital
    
    # Alpha (exceso sobre buy & hold)
    alpha = strategy_return - buyhold_return
    
    return {
        'strategy_return': strategy_return,
        'buyhold_return': buyhold_return,
        'alpha': alpha,
        'strategy_capitals': capitals,
        'buyhold_capitals': buyhold_capitals
    }


def print_report(metrics, comparison, results):
    """Imprime reporte profesional."""
    
    print("\n" + "=" * 70)
    print("                    📊 BACKTEST REPORT")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         PERFORMANCE SUMMARY                          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Total Return:          {metrics['total_return']*100:>+10.2f}%                              │")
    print(f"│  Buy & Hold Return:     {comparison['buyhold_return']*100:>+10.2f}%                              │")
    print(f"│  Alpha (vs B&H):        {comparison['alpha']*100:>+10.2f}%                              │")
    print(f"│  Final Capital:         ${metrics['final_capital']:>10,.2f}                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         RISK METRICS                                 │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}                                  │")
    print(f"│  Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}                                  │")
    print(f"│  Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}                                  │")
    print(f"│  Max Drawdown:          {metrics['max_drawdown']*100:>10.2f}%                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         TRADE STATISTICS                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Total Trades:          {metrics['n_trades']:>10.0f}                                  │")
    print(f"│  Win Rate:              {metrics['win_rate']*100:>10.1f}%                              │")
    print(f"│  Profit Factor:         {metrics['profit_factor']:>10.2f}                                  │")
    print(f"│  Avg Trade Return:      {metrics['avg_trade']*100:>10.3f}%                              │")
    print(f"│  Exposure:              {metrics['exposure']*100:>10.1f}%                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                         EXTREMES                                     │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Best Period:           {metrics['best_period']:>+10.2f}%                              │")
    print(f"│  Worst Period:          {metrics['worst_period']:>+10.2f}%                              │")
    print(f"│  Max Consec. Wins:      {metrics['max_consec_wins']:>10.0f}                                  │")
    print(f"│  Max Consec. Losses:    {metrics['max_consec_losses']:>10.0f}                                  │")
    print("└─────────────────────────────���───────────────────────────────────────┘")
    
    # Distribución de regímenes
    regimes = results['regimes']
    regime_names = ['Trend Up', 'Trend Down', 'Ranging', 'High Vol']
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                      REGIME DISTRIBUTION                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    for i, name in enumerate(regime_names):
        pct = np.mean(regimes == i) * 100
        bar = "█" * int(pct / 2)
        print(f"│  {name:<12}: {pct:>5.1f}% {bar:<25}            │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Evaluación final
    print("\n" + "=" * 70)
    if metrics['sharpe_ratio'] > 2 and metrics['total_return'] > 0.1:
        print("                    ✅ ESTRATEGIA EXCELENTE")
        print("        Listo para paper trading con monitoreo cercano")
    elif metrics['sharpe_ratio'] > 1 and metrics['total_return'] > 0:
        print("                    🟡 ESTRATEGIA PROMETEDORA")
        print("        Considerar optimización de hiperparámetros")
    else:
        print("                    🔴 ESTRATEGIA NECESITA MEJORAS")
        print("        Revisar arquitectura o datos")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("              🔬 BACKTEST PROFESIONAL - MULTI-TIMEFRAME")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # Cargar datos
    logger.info("📥 Cargando datos...")
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    # Indicadores
    df_1h = calculate_all_indicators(df_1h)
    df_4h = calculate_all_indicators(df_4h)
    df_1d = calculate_all_indicators(df_1d)
    
    # ATR
    if 'atr' not in df_4h.columns:
        high, low, close = df_4h['high'].values, df_4h['low'].values, df_4h['close'].values
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        df_4h['atr'] = pd.Series(tr).rolling(14).mean().values
    
    # Cargar modelos
    detector, agent = load_models(device)
    
    # Crear dataset
    logger.info("📦 Creando dataset...")
    dataset = MultiTimeframeDataset(df_1h, df_4h, df_1d)
    
    # Solo usar datos de TEST (último 20%)
    test_start = int(len(dataset) * 0.8)
    test_indices = list(range(test_start, len(dataset)))
    
    logger.info(f"   Test samples: {len(test_indices)}")
    
    # Generar embeddings para test
    logger.info("🧠 Generando embeddings...")
    
    detector.eval()
    embeddings_list = []
    pred_4h_list = []
    pred_12h_list = []
    pred_24h_list = []
    regimes_list = []
    prices_list = []
    
    with torch.no_grad():
        for idx in test_indices:
            sample = dataset[idx]
            x_1d = sample['x_1d'].unsqueeze(0).to(device)
            x_4h = sample['x_4h'].unsqueeze(0).to(device)
            x_1h = sample['x_1h'].unsqueeze(0).to(device)
            
            outputs = detector(x_1d, x_4h, x_1h)
            
            embeddings_list.append(outputs['embedding'].cpu().numpy())
            pred_4h_list.append(outputs['pred_4h'].cpu().numpy())
            pred_12h_list.append(outputs['pred_12h'].cpu().numpy())
            pred_24h_list.append(outputs['pred_24h'].cpu().numpy())
            regimes_list.append(outputs['regime_logits'].argmax(dim=-1).cpu().numpy())
            prices_list.append(sample['price'].numpy())
    
    embeddings = np.vstack(embeddings_list)
    pred_4h = np.vstack(pred_4h_list)
    pred_12h = np.vstack(pred_12h_list)
    pred_24h = np.vstack(pred_24h_list)
    regimes = np.concatenate(regimes_list)
    prices = np.array(prices_list)
    
    # ATRs para test
    atrs = df_4h['atr'].values[-len(prices):]
    atrs = np.nan_to_num(atrs, nan=np.nanmean(df_4h['atr'].values))
    
    # Crear entorno de test
    config = AdvancedTradingConfig()
    env = AdvancedTradingEnv(
        embeddings, pred_4h, pred_12h, pred_24h, regimes, prices, atrs, config
    )
    
    # Ejecutar backtest
    logger.info("🔬 Ejecutando backtest...")
    results = run_backtest_simulation(agent, env, deterministic=True)
    
    # Calcular métricas
    metrics = calculate_metrics(results, prices)
    comparison = compare_with_buyhold(results['capitals'], prices, config.initial_capital)
    
    # Imprimir reporte
    print_report(metrics, comparison, results)
    
    # Guardar resultados
    results_path = root_dir / "artifacts/backtest_results.npz"
    np.savez(
        results_path,
        capitals=results['capitals'],
        positions=results['positions'],
        returns=results['returns'],
        prices=prices[:len(results['capitals'])],
        **metrics
    )
    logger.info(f"\n💾 Resultados guardados: {results_path}")
    
    print(f"\n🎯 Siguiente paso: python scripts/05_optimize.py")


if __name__ == "__main__":
    main()
