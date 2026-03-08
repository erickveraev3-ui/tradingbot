import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3


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

            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)

        )

    def forward(self,x):

        return self.net(x)


def train_model(df, candidate_col, label_col, model_name):

    logger.info(f"Training {model_name}")

    df = df[df[candidate_col] == 1].copy()

    y = (df[label_col] == 1).astype(int).values

    feature_cols = [
        c for c in df.columns
        if c not in [
            "timestamp",
            "candidate_long",
            "candidate_short",
            "tb_long_label",
            "tb_short_label",
            "tb_long_return",
            "tb_short_return",
            "tb_long_hit_bar",
            "tb_short_hit_bar"
        ]
    ]

    X = df[feature_cols].values

    split = int(len(X)*0.8)

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]

    scaler = RobustScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pos_weight = (len(y_train)-y_train.sum())/max(y_train.sum(),1)

    logger.info(f"Positive rate: {y_train.mean():.4f}")
    logger.info(f"pos_weight: {pos_weight:.2f}")

    train_ds = TradingDataset(X_train,y_train)
    val_ds = TradingDataset(X_val,y_val)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE)

    model = MetaModel(X.shape[1]).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).to(DEVICE)
    )

    opt = torch.optim.Adam(model.parameters(),lr=LR)

    for epoch in range(EPOCHS):

        model.train()

        losses=[]

        for Xb,yb in train_loader:

            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(1)

            logits = model(Xb)

            loss = loss_fn(logits,yb)

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

                logits = model(Xb)

                prob = torch.sigmoid(logits).cpu().numpy().flatten()

                preds.extend(prob)
                trues.extend(yb.numpy())

        auc = roc_auc_score(trues,preds)
        ap = average_precision_score(trues,preds)

        logger.info(
            f"epoch {epoch} loss {np.mean(losses):.4f} auc {auc:.4f} ap {ap:.4f}"
        )

    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pt")

    logger.info(f"Saved {model_name}")



def main():

    logger.info("Loading dataset")

    df = pd.read_csv(DATA_PATH)

    train_model(
        df,
        candidate_col="candidate_long",
        label_col="tb_long_label",
        model_name="meta_model_long"
    )

    train_model(
        df,
        candidate_col="candidate_short",
        label_col="tb_short_label",
        model_name="meta_model_short"
    )


if __name__ == "__main__":

    main()

