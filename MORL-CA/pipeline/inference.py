import argparse
import numpy as np
import pandas as pd
import os
import onnxruntime as ort
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

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
    return problem, kpi_normalizer

def calculate_indicators(pareto_O, pareto_C, kpi_normalizer, ref_pareto=None, ref_point=None):
    feasible_mask = np.all(pareto_C <= 1e-6, axis=1)
    pareto_O_feasible = pareto_O[feasible_mask] if np.any(feasible_mask) else pareto_O

    if len(pareto_O_feasible) == 0:
        return float('inf'), 0.0

    if ref_pareto is None:
        ref_data = np.load('global_pareto_front.npz')
        ref_pareto = ref_data['pareto_front']
        ref_point = ref_data['ref_point']

    if len(ref_pareto) == 0:
        return float('inf'), 0.0

    ref_norm = kpi_normalizer.transform(ref_pareto)
    ref_point_norm = kpi_normalizer.transform(ref_point.reshape(1, -1))[0]

    pareto_norm = pareto_O_feasible

    igd = float('inf')
    hv = 0.0

    try:
        igd_indicator = IGD(ref_norm)
        igd = igd_indicator.do(pareto_norm)
    except Exception as e:
        print(f"IGD calculation error: {e}")

    try:
        hv_indicator = Hypervolume(ref_point=ref_point_norm)
        hv = hv_indicator.do(pareto_norm)
    except Exception as e:
        print(f"HV calculation error: {e}")

    return igd, hv

def run_inference_for_restric_val(model_path, restric_val, start_points_sample, max_steps, max_size,
                                  net_arch_hidden_dim, aow_hidden_dim, penalty_coeff, buffer_class,
                                  aow_model_path, ge_from_file, result_dir='./aow_inference_results'):
    print(f"\n{'='*60}")
    print(f"Running inference for restric_val = {restric_val}")
    print(f"{'='*60}")

    problem, kpi_normalizer = build_CA_Problem(restric_val)
    print(f"Problem created: state_dim={problem.state_dim}, obj_dim={problem.obj_dim}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"Loading trained model from {model_path}")

    sac = SAC_Weighted_Arch(
        problem,
        net_arch_hidden_dim=net_arch_hidden_dim,
        aow_hidden_dim=aow_hidden_dim,
        penalty_coeff=penalty_coeff,
        buffer_class=buffer_class,
        tensorboard_log='./tensorboard_logs/aow_test/',
        aow_model_path=aow_model_path
    )
    sac.load(model_path)
    print("Model loaded successfully")

    print(f"\nRunning optimization: start_points_sample={start_points_sample}, "
          f"max_steps={max_steps}, max_size={max_size}")
    pareto_X, pareto_O, pareto_C = sac.optimize(
        start_points_sample=start_points_sample,
        max_steps=max_steps,
        max_size=max_size,
        GE_from_file=ge_from_file
    )
    print(f"Optimization result: {len(pareto_X)} Pareto solutions found")

    igd, hv = calculate_indicators(pareto_O, pareto_C, kpi_normalizer)
    print(f"  IGD: {igd:.4f}, HV: {hv:.4f}")

    KPI = np.array([get_kpi_from_onnx(x) for x in pareto_X])

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f'restric_val{restric_val}.npz')
    np.savez(result_path, X=pareto_X, O=pareto_O, C=pareto_C, KPI=KPI, igd=igd, hv=hv)
    print(f"Results saved to {result_path}")

    return pareto_X, pareto_O, pareto_C, igd, hv

def run_inference(model_path, start_points_sample=5, max_steps=100, max_size=20000,
                  net_arch_hidden_dim=32, aow_hidden_dim=64, penalty_coeff=1e3,
                  buffer_class=BufferClass.FIFO, aow_model_path='./AOW_Model/aow_network.pth',
                  ge_dir='./CA_Optimization/NSGAII', result_dir='./aow_inference_results'):

    results = {}
    restric_vals = range(86, 120)

    for restric_val in restric_vals:
        ge_from_file = f'{ge_dir}/restric_val{restric_val}.txt'

        try:
            pareto_X, pareto_O, pareto_C, igd, hv = run_inference_for_restric_val(
                model_path=model_path,
                restric_val=restric_val,
                start_points_sample=start_points_sample,
                max_steps=max_steps,
                max_size=max_size,
                net_arch_hidden_dim=net_arch_hidden_dim,
                aow_hidden_dim=aow_hidden_dim,
                penalty_coeff=penalty_coeff,
                buffer_class=buffer_class,
                aow_model_path=aow_model_path,
                ge_from_file=ge_from_file,
                result_dir=result_dir
            )
            results[restric_val] = {
                'X': pareto_X,
                'O': pareto_O,
                'C': pareto_C,
                'igd': igd,
                'hv': hv
            }
        except Exception as e:
            print(f"Error for restric_val={restric_val}: {e}")
            results[restric_val] = None

    return results

def main():
    parser = argparse.ArgumentParser(description='Run inference with pre-trained SAC_Weighted_Arch')

    parser.add_argument('--model_path', type=str, default='./models/sac_weighted_trained.zip',
                        help='Path to trained SAC_Weighted_Arch model')

    parser.add_argument('--ge_dir', type=str, default='./CA_Optimization/NSGAII',
                        help='Directory containing GE results for each restric_val')

    parser.add_argument('--net_arch_hidden_dim', type=int, default=64, help='Hidden dimension for network architecture')
    parser.add_argument('--aow_hidden_dim', type=int, default=64, help='Hidden dimension for AOW network')
    parser.add_argument('--penalty_coeff', type=float, default=1e3, help='Penalty coefficient for constraints')
    parser.add_argument('--buffer_class', type=str, default='FIFO', choices=['FIFO', 'PER'], help='Replay buffer class')
    parser.add_argument('--aow_model_path', type=str, default='./AOW_Model/aow_network.pth', help='Path to AOW model')

    parser.add_argument('--start_points_sample', type=int, default=1000, help='Number of start points for optimization')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per solution')
    parser.add_argument('--max_size', type=int, default=20000, help='Maximum size of Pareto archive')

    args = parser.parse_args()

    buffer_class_map = {'FIFO': BufferClass.FIFO, 'PER': BufferClass.PER}
    buffer_class = buffer_class_map[args.buffer_class]

    print("=" * 60)
    print("AOW-Adapted SAC Inference Test")
    print("=" * 60)
    print(f"Using model: {args.model_path}")
    print(f"GE directory: {args.ge_dir}")
    print(f"Restric_vals: 86 to 119")

    try:
        results = run_inference(
            model_path=args.model_path,
            start_points_sample=args.start_points_sample,
            max_steps=args.max_steps,
            max_size=args.max_size,
            net_arch_hidden_dim=args.net_arch_hidden_dim,
            aow_hidden_dim=args.aow_hidden_dim,
            penalty_coeff=args.penalty_coeff,
            buffer_class=buffer_class,
            aow_model_path=args.aow_model_path,
            ge_dir=args.ge_dir
        )

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for restric_val, result in results.items():
            if result is not None:
                print(f"restric_val={restric_val}: {len(result['X'])} solutions, IGD={result['igd']:.4f}, HV={result['hv']:.4f}")
            else:
                print(f"restric_val={restric_val}: FAILED")

        print("\nInference completed successfully!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
