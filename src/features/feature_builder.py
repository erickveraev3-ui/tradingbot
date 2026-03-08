import pandas as pd
import numpy as np

from .indicators import calculate_all_indicators
from .indicators_pro import calculate_pro_indicators


class FeatureBuilder:
    """
    Construye features para el modelo.
    BTC = activo objetivo
    ETH y SOL = contexto del mercado crypto
    """

    def __init__(self):
        pass

    def build(self, btc: pd.DataFrame, eth: pd.DataFrame, sol: pd.DataFrame) -> pd.DataFrame:

        btc = btc.copy()
        eth = eth.copy()
        sol = sol.copy()

        # =========================
        # INDICADORES
        # =========================

        btc = calculate_all_indicators(btc)
        btc = calculate_pro_indicators(btc)

        eth = calculate_all_indicators(eth)
        sol = calculate_all_indicators(sol)

        df = btc.copy()

        # =========================
        # CROSS ASSET FEATURES
        # =========================

        df["eth_return_1h"] = eth["close"].pct_change()
        df["sol_return_1h"] = sol["close"].pct_change()

        df["eth_return_4h"] = eth["close"].pct_change(4)
        df["sol_return_4h"] = sol["close"].pct_change(4)

        df["btc_eth_ratio"] = btc["close"] / eth["close"]
        df["btc_sol_ratio"] = btc["close"] / sol["close"]

        df["btc_eth_divergence"] = df["return_1"] - df["eth_return_1h"]
        df["btc_sol_divergence"] = df["return_1"] - df["sol_return_1h"]

        # volatilidad relativa
        df["btc_volatility"] = df["return_1"].rolling(24).std()
        df["eth_volatility"] = df["eth_return_1h"].rolling(24).std()

        df["volatility_ratio"] = df["btc_volatility"] / (df["eth_volatility"] + 1e-8)

        # =========================
        # TARGETS
        # =========================

        df["target_return_1h"] = df["close"].shift(-1) / df["close"] - 1
        df["target_return_4h"] = df["close"].shift(-4) / df["close"] - 1

        df["target_direction"] = (df["target_return_1h"] > 0).astype(int)

        df = df.dropna().reset_index(drop=True)

        return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
