import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 64
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TradingDataset(Dataset):

    def __init__(self, X, y):

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class MetaModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,1),
            nn.Sigmoid()

        )

    def forward(self,x):

        return self.net(x)


def main():

    logger.info("Cargando dataset")

    df = pd.read_csv(DATA_PATH)

    feature_cols = [

        c for c in df.columns
        if c not in [
            "timestamp",
            "tb_long_label",
            "tb_short_label",
            "tb_long_return",
            "tb_short_return",
            "tb_long_hit_bar",
            "tb_short_hit_bar"
        ]
    ]

    X = df[feature_cols].values

    y_long = (df["tb_long_label"] == 1).astype(int).values
    y_short = (df["tb_short_label"] == 1).astype(int).values

    logger.info(f"Features: {len(feature_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_long, test_size=0.2, shuffle=False
    )

    train_ds = TradingDataset(X_train,y_train)
    val_ds = TradingDataset(X_val,y_val)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE)

    model = MetaModel(X.shape[1]).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):

        model.train()

        losses = []

        for Xb,yb in train_loader:

            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(1)

            pred = model(Xb)

            loss = loss_fn(pred,yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        model.eval()

        preds=[]
        trues=[]

        with torch.no_grad():

            for Xb,yb in val_loader:

                Xb = Xb.to(DEVICE)

                pred = model(Xb).cpu().numpy().flatten()

                preds.extend(pred)
                trues.extend(yb.numpy())

        auc = roc_auc_score(trues,preds)
        ap = average_precision_score(trues,preds)

        logger.info(
            f"epoch {epoch} loss {np.mean(losses):.4f} auc {auc:.4f} ap {ap:.4f}"
        )

    torch.save(model.state_dict(), MODEL_DIR / "meta_model_long.pt")

    logger.info("Modelo guardado")


if __name__ == "__main__":

    main()
