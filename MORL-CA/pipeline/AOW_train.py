import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import onnxruntime as ort
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

session = ort.InferenceSession('CA_model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def get_CA_data():
    import pandas as pd
    y_cols = ['daily_prod', 'curr_eff', 'djdh']
    train_data = pd.read_csv("CA_train.csv")
    test_data = pd.read_csv("CA_test.csv")
    train_x = train_data.drop(columns=y_cols).values
    train_y = train_data[y_cols].values
    test_x = test_data.drop(columns=y_cols).values
    test_y = test_data[y_cols].values
    return train_x, train_y, test_x, test_y

class KPINormalizer:
    def __init__(self):
        self.scaler = None

    def fit(self, data: np.ndarray):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)

    def transform(self, data: np.ndarray):
        return self.scaler.transform(data)

def get_kpi_from_onnx(inputs):
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    kpis_res = session.run([output_name], {input_name: inputs})[0]
    kpis_res = kpis_res.transpose()
    return kpis_res[0]

class AOWNet(nn.Module):
    def __init__(self, input_dim: int, n_obj: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_obj)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        weights = torch.softmax(logits, dim=-1)
        return weights

def compute_target_weights(objs, beta=5.0, eps=1e-8):
    objs = np.asarray(objs, dtype=np.float32)

    scores = 1.0 - objs
    scores = 0.9 * scores + 0.1 / scores.shape[1]
    scores = np.clip(scores, eps, None)

    logits = beta * scores
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return weights

def sample_states_from_all_restric_vals(N_per_val=1000, restric_vals=None, kpi_normalizer=None, max_trials=100000):
    if restric_vals is None:
        restric_vals = range(86, 120)

    train_x, train_y, test_x, test_y = get_CA_data()

    params_range_base = []
    for i in range(train_x.shape[1]):
        low = np.min(train_x[:, i])
        high = np.max(train_x[:, i])
        params_range_base.append([low, high])

    X_all = []
    objs_all = []

    for restric_val in restric_vals:
        params_range = [list(b) for b in params_range_base]
        nod = 8
        for i in range(nod):
            params_range[i*7][0] = restric_val/nod - 0.5
            params_range[i*7][1] = restric_val/nod + 0.5

        X_valid = []
        objs_valid = []
        trials = 0

        while len(X_valid) < N_per_val and trials < max_trials:
            trials += 1
            x = np.random.uniform(
                low=[b[0] for b in params_range],
                high=[b[1] for b in params_range]
            )
            try:
                kpis = get_kpi_from_onnx(x)
                objs = kpi_normalizer.transform(kpis.reshape(1, -1))[0]
                if np.any(objs <= 0) or np.any(objs >= 1.0):
                    continue
                X_valid.append(x)
                objs_valid.append(objs)
            except:
                continue

        X_all.extend(X_valid)
        objs_all.extend(objs_valid)
        print(f"restric_val={restric_val}: sampled {len(X_valid)} valid states")

    return np.array(X_all), np.array(objs_all)

def train_aow_network(
    n_samples_per_val=1000,
    restric_vals=None,
    hidden_dim=64,
    epochs=500,
    batch_size=64,
    lr=1e-3,
    random_state=0,
    save_path='./AOW_Model/aow_network.pth'
):
    print(f"Training AOW network with {n_samples_per_val} samples per restric_val...")

    if restric_vals is None:
        restric_vals = list(range(86, 120))

    train_x, train_y, test_x, test_y = get_CA_data()

    kpi_normalizer = KPINormalizer()
    kpi_normalizer.fit(np.vstack([train_y, test_y]))

    X, objs = sample_states_from_all_restric_vals(
        N_per_val=n_samples_per_val,
        restric_vals=restric_vals,
        kpi_normalizer=kpi_normalizer,
        max_trials=100000
    )
    print(f"Total sampled {len(X)} states")

    Y = compute_target_weights(objs, beta=5.0)
    print(f"Computed target weights, shape: {Y.shape}")

    X_train, X_test, Y_train, Y_test, objs_train, objs_test = train_test_split(
        X, Y, objs, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_s)
    Y_train_t = torch.FloatTensor(Y_train)
    X_test_t = torch.FloatTensor(X_test_s)
    Y_test_t = torch.FloatTensor(Y_test)

    input_dim = X_train_s.shape[1]
    n_obj = Y.shape[1]
    model = AOWNet(input_dim=input_dim, n_obj=n_obj, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t)
                test_loss = criterion(test_pred, Y_test_t).item()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(dataloader):.4f}, Test Loss: {test_loss:.4f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        train_mse = criterion(train_pred, Y_train_t).item()
        test_pred = model(X_test_t)
        test_mse = criterion(test_pred, Y_test_t).item()
    print(f"Final Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'input_dim': input_dim,
        'n_obj': n_obj,
        'hidden_dim': hidden_dim
    }, save_path)
    print(f"Saved AOW model to {save_path}")

    return model, scaler, X, objs, Y

def predict_and_validate(model_state_dict, scaler_data, X, objs, margin=0.0):
    input_dim = model_state_dict['input_dim']
    n_obj = model_state_dict['n_obj']
    hidden_dim = model_state_dict['hidden_dim']

    model = AOWNet(input_dim=input_dim, n_obj=n_obj, hidden_dim=hidden_dim)
    model.load_state_dict(model_state_dict['model_state_dict'])
    model.eval()

    mean = model_state_dict['scaler_mean']
    scale = model_state_dict['scaler_scale']
    X_s = (X - mean) / scale
    X_t = torch.FloatTensor(X_s)

    with torch.no_grad():
        Y_pred = model(X_t).numpy()

    Y_pred = np.clip(Y_pred, 1e-9, None)
    Y_pred = Y_pred / Y_pred.sum(axis=1, keepdims=True)

    obj0_better = (objs[:, 0] <= np.minimum(objs[:, 1], objs[:, 2]) - margin)
    obj1_better = (objs[:, 1] <= np.minimum(objs[:, 0], objs[:, 2]) - margin)
    obj2_better = (objs[:, 2] <= np.minimum(objs[:, 0], objs[:, 1]) - margin)

    labels = np.full(len(objs), -1, dtype=int)
    for i in range(len(objs)):
        if obj0_better[i]:
            labels[i] = 0
        elif obj1_better[i]:
            labels[i] = 1
        elif obj2_better[i]:
            labels[i] = 2

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    idx2 = np.where(labels == 2)[0]

    print(f"\nClassification counts:")
    print(f"  obj0_best (region_1): {len(idx0)}")
    print(f"  obj1_best (region_2): {len(idx1)}")
    print(f"  obj2_best (region_3): {len(idx2)}")

    mean_w0 = Y_pred[idx0].mean(axis=0) if len(idx0) > 0 else np.array([np.nan]*3)
    mean_w1 = Y_pred[idx1].mean(axis=0) if len(idx1) > 0 else np.array([np.nan]*3)
    mean_w2 = Y_pred[idx2].mean(axis=0) if len(idx2) > 0 else np.array([np.nan]*3)

    print(f"\nMean weights for groups:")
    print(f"  Group obj0_best: {mean_w0}")
    print(f"  Group obj1_best: {mean_w1}")
    print(f"  Group obj2_best: {mean_w2}")

    cmap = "viridis"
    marker_size = 25
    alpha = 0.9

    fig = plt.figure(figsize=(18, 5))
    groups = [(idx0, 0), (idx1, 1), (idx2, 2)]
    titles = ["Obj1-best (region_1)", "Obj2-best (region_2)", "Obj3-best (region_3)"]

    for i, (idxs, target_w) in enumerate(groups):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        if len(idxs) == 0:
            ax.text(0.5, 0.5, 0.5, "No points", horizontalalignment='center')
            ax.set_title(titles[i] + "  (count=0)")
            continue

        xs = objs[idxs, 0]
        ys = objs[idxs, 1]
        zs = objs[idxs, 2]
        colors = Y_pred[idxs, target_w]

        sc = ax.scatter(xs, ys, zs, c=colors, cmap=cmap, s=marker_size, alpha=alpha)
        ax.set_xlabel("obj0")
        ax.set_ylabel("obj1")
        ax.set_zlabel("obj2")
        ax.set_title(f"{titles[i]}  (count={len(idxs)})")
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label(f"w{target_w}")

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"aow_validation.{ext}", format=ext, dpi=300, bbox_inches="tight")
    plt.show()

    groups_means = np.vstack([mean_w0, mean_w1, mean_w2])
    labels_groups = [r"$\mathrm{region}_1$", r"$\mathrm{region}_2$", r"$\mathrm{region}_3$"]
    x = np.arange(3)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
    width = 0.2
    for i in range(3):
        ax2.bar(x + (i-1)*width, groups_means[i], width=width, label=labels_groups[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$w_{j}$" for j in range(3)], fontsize=16)
    ax2.set_ylabel("Mean Predicted Weight", fontsize=16)
    ax2.legend()
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(f"aow_mean_weights.{ext}", format=ext, dpi=300, bbox_inches="tight")
    plt.show()

    return Y_pred, labels, mean_w0, mean_w1, mean_w2

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train AOW network')
    parser.add_argument('--n_per_val', type=int, default=1000, help='Samples per restric_val')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_path', type=str, default='./AOW_Model/aow_network.pth', help='Save path')

    args = parser.parse_args()

    model, scaler, X, objs, Y = train_aow_network(
        n_samples_per_val=args.n_per_val,
        restric_vals=list(range(86, 120)),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )

    checkpoint = torch.load(args.save_path, weights_only=False)
    predict_and_validate(checkpoint, None, X, objs, margin=0.0)