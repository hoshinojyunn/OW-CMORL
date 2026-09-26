import argparse
import numpy as np
import pandas as pd
import os
import onnxruntime as ort
import torch
from arch.SAC_arch import *

np.random.seed(0)

def get_CA_data():
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

session = ort.InferenceSession('CA_model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def get_kpi_from_onnx(inputs):
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    kpis_res = session.run([output_name], {input_name: inputs})[0]
    kpis_res = kpis_res.transpose()
    return kpis_res[0]

def build_CA_Problem(restric_val=86):
    train_x, train_y, test_x, test_y = get_CA_data()

    kpi_normalizer = KPINormalizer()
    kpi_normalizer.fit(np.vstack([train_y, test_y]))

    def obj1(inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        kpis_res = get_kpi_from_onnx(inputs)
        kpis_res = kpi_normalizer.transform(kpis_res.reshape(1, -1))[0]
        return -kpis_res[0]

    def obj2(inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        kpis_res = get_kpi_from_onnx(inputs)
        kpis_res[1] = np.clip(kpis_res[1], 0, 100)
        kpis_res = kpi_normalizer.transform(kpis_res.reshape(1, -1))[0]
        return -kpis_res[1]

    def obj3(inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        kpis_res = get_kpi_from_onnx(inputs)
        kpis_res = kpi_normalizer.transform(kpis_res.reshape(1, -1))[0]
        return kpis_res[2]

    def current_constraint(inputs):
        nod = 8
        values = [inputs[i*7] for i in range(nod)]
        return max(0, sum(values) - restric_val)

    def current_eff_constraint(inputs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        kpis_res = get_kpi_from_onnx(inputs)
        return max(0, kpis_res[1] - 100)

    params_range = []
    for i in range(train_x.shape[1]):
        low = np.min(train_x[:, i])
        high = np.max(train_x[:, i])
        params_range.append([low, high])

    nod = 8
    for i in range(nod):
        params_range[i*7][0] = restric_val/nod - 0.5
        params_range[i*7][1] = restric_val/nod + 0.5

    constraints_list = [current_constraint, current_eff_constraint]
    problem = MOO_Problem([obj1, obj2, obj3], constraints_list, bounds=params_range)
    return problem

def load_aow_hidden_dim(aow_model_path):
    if not os.path.exists(aow_model_path):
        raise FileNotFoundError(f"AOW model not found at {aow_model_path}. Please train AOW first.")
    checkpoint = torch.load(aow_model_path, weights_only=False)
    return checkpoint['hidden_dim']

def train_sac_weighted(
    net_arch_hidden_dim=32,
    penalty_coeff=1e3,
    buffer_class=BufferClass.FIFO,
    tensorboard_log='./tensorboard_logs/sac_weighted_train/',
    aow_model_path='./AOW_Model/aow_network.pth',
    initial_population_size=10,
    fine_tune_steps=100,
    batch_size=64,
    use_nsga2=False,
    model_save_path='./models/sac_weighted_trained.zip'
):

    restric_vals = list(range(86, 120))

    aow_model_path = os.path.abspath(aow_model_path)
    aow_hidden_dim = load_aow_hidden_dim(aow_model_path)
    print(f"Loaded AOW model from {aow_model_path}, hidden_dim={aow_hidden_dim}")

    initial_problem = build_CA_Problem(restric_vals[0])
    print(f"\nProblem created: state_dim={initial_problem.state_dim}, obj_dim={initial_problem.obj_dim}")

    os.makedirs(tensorboard_log, exist_ok=True)
    sac = SAC_Weighted_Arch(
        initial_problem,
        net_arch_hidden_dim=net_arch_hidden_dim,
        aow_hidden_dim=aow_hidden_dim,
        penalty_coeff=penalty_coeff,
        buffer_class=buffer_class,
        tensorboard_log=tensorboard_log,
        aow_model_path=aow_model_path
    )
    print("SAC_Weighted_Arch created with AOW")

    print(f"\nStarting multi-problem training:")
    print(f"  restric_vals: {restric_vals[0]}-{restric_vals[-1]} ({len(restric_vals)} problems)")
    print(f"  initial_population_size: {initial_population_size}")
    print(f"  fine_tune_steps: {fine_tune_steps}")
    print(f"  use_nsga2: {use_nsga2}")

    np.random.shuffle(restric_vals)

    for rv_idx, restric_val in enumerate(restric_vals):
        problem = build_CA_Problem(restric_val)
        sac.train_env.problem = problem
        sac.train_env.penalty_coeff = penalty_coeff

        sac.learn(
            initial_population_size=initial_population_size,
            steps_per_episode=fine_tune_steps,
            batch_size=batch_size,
            use_nsga2=use_nsga2
        )

        if (rv_idx + 1) % 10 == 0:
            print(f"  Completed {rv_idx + 1}/{len(restric_vals)} problems")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    sac.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    return sac, model_save_path

def main():
    parser = argparse.ArgumentParser(description='Train SAC_Weighted_Arch with pre-trained AOW across all restric_vals (86-119)')

    parser.add_argument('--net_arch_hidden_dim', type=int, default=64, help='Hidden dimension for network architecture')
    parser.add_argument('--penalty_coeff', type=float, default=1e3, help='Penalty coefficient for constraints')
    parser.add_argument('--buffer_class', type=str, default='FIFO', choices=['FIFO', 'PER'], help='Replay buffer class')
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs/sac_weighted_train/', help='Tensorboard log directory')

    parser.add_argument('--aow_model_path', type=str, default='./AOW_Model/aow_network.pth', help='Path to pre-trained AOW model')

    parser.add_argument('--initial_population_size', type=int, default=1000, help='Number of start points per episode')
    parser.add_argument('--fine_tune_steps', type=int, default=100, help='Number of fine-tune steps per start point')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--use_nsga2', action='store_true', help='Use NSGA2 for initial population (default: False)')

    parser.add_argument('--model_save_path', type=str, default='./models/sac_weighted_trained.zip', help='Path to save trained model')

    args = parser.parse_args()

    buffer_class_map = {'FIFO': BufferClass.FIFO, 'PER': BufferClass.PER}
    buffer_class = buffer_class_map[args.buffer_class]

    print("=" * 60)
    print("SAC_Weighted_Arch Multi-Problem Training")
    print("=" * 60)
    print(f"Training parameters:")
    print(f"  restric_vals: 86-119 (34 problems, fixed)")
    print(f"  net_arch_hidden_dim: {args.net_arch_hidden_dim}")
    print(f"  penalty_coeff: {args.penalty_coeff}")
    print(f"  buffer_class: {args.buffer_class}")
    print(f"  aow_model_path: {args.aow_model_path}")
    print(f"  initial_population_size: {args.initial_population_size}")
    print(f"  fine_tune_steps: {args.fine_tune_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  use_nsga2: {args.use_nsga2}")

    sac, save_path = train_sac_weighted(
        net_arch_hidden_dim=args.net_arch_hidden_dim,
        penalty_coeff=args.penalty_coeff,
        buffer_class=buffer_class,
        tensorboard_log=args.tensorboard_log,
        aow_model_path=args.aow_model_path,
        initial_population_size=args.initial_population_size,
        fine_tune_steps=args.fine_tune_steps,
        batch_size=args.batch_size,
        use_nsga2=args.use_nsga2,
        model_save_path=args.model_save_path
    )

    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()