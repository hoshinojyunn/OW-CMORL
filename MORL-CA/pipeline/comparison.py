import argparse
import numpy as np
import pandas as pd
import os
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

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

def load_npz_results(algo_dir, restric_vals):
    all_X = []
    all_KPI = []
    all_C = []

    for val in restric_vals:
        filepath = os.path.join(algo_dir, f'restric_val{val}.npz')
        if os.path.exists(filepath):
            data = np.load(filepath)
            all_X.append(data['X'])
            all_KPI.append(data['KPI'])
            if 'C' in data:
                all_C.append(data['C'])
            else:
                all_C.append(np.zeros((len(data['X']), 0)))
        else:
            print(f"Warning: {filepath} not found")

    return all_X, all_KPI, all_C

def sample_pareto(Y, max_size=5000):
    if len(Y) <= max_size:
        return Y
    idx = np.linspace(0, Y.shape[0] - 1, max_size).astype(int)
    return Y[idx]

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

def run_comparison(optim_dir='./CA_Optimization',
                   restric_vals=None):
    if restric_vals is None:
        restric_vals = list(range(86, 120))

    algo_folders = []
    for item in os.listdir(optim_dir):
        item_path = os.path.join(optim_dir, item)
        if os.path.isdir(item_path):
            algo_folders.append(item)

    algo_folders = sorted(algo_folders)
    print(f"Found {len(algo_folders)} algorithm folders: {algo_folders}")

    _, train_y, _, test_y = get_CA_data()
    kpi_normalizer = KPINormalizer()
    kpi_normalizer.fit(np.vstack([train_y, test_y]))

    ref_data_path = os.path.join(os.path.dirname(__file__), 'global_pareto_front.npz')
    if os.path.exists(ref_data_path):
        ref_data = np.load(ref_data_path)
        ref_pareto = ref_data['pareto_front']
        ref_point = ref_data['ref_point']
    else:
        ref_pareto = None
        ref_point = None
        print("Warning: global_pareto_front.npz not found, using first algorithm as reference")

    summary = {}

    for algo in algo_folders:
        algo_path = os.path.join(optim_dir, algo)
        print(f"\n{'='*60}")
        print(f"Processing algorithm: {algo}")
        print(f"{'='*60}")

        algo_igd = []
        algo_hv = []

        for val in restric_vals:
            npz_file = os.path.join(algo_path, f'restric_val{val}.npz')

            if not os.path.exists(npz_file):
                continue

            try:
                data = np.load(npz_file)
                if 'pareto_kpi' in data:
                    KPI = data['pareto_kpi']
                elif 'KPI' in data:
                    KPI = data['KPI']
                else:
                    print(f"No KPI data found in {npz_file}")
                    continue

                if len(KPI) == 0:
                    continue

                C = data['C'] if 'C' in data else np.zeros((len(KPI), 0))

                KPI_sampled = sample_pareto(KPI, max_size=5000)
                C_sampled = sample_pareto(C, max_size=5000) if len(C) > 0 else np.zeros((len(KPI_sampled), 0))

                if ref_pareto is None and algo == algo_folders[0]:
                    ref_pareto = KPI_sampled.copy()
                    ref_point = np.max(KPI_sampled, axis=0) + 0.1
                    print(f"Using {algo} as reference front")

                igd_indicator = IGD(ref_pareto)
                igd = igd_indicator.do(KPI_sampled)

                hv_indicator = Hypervolume(ref_point=ref_point)
                hv = hv_indicator.do(KPI_sampled)

                algo_igd.append(igd)
                algo_hv.append(hv)

                print(f"restric_val={val}: IGD={igd:.4f}, HV={hv:.4f}")
            except Exception as e:
                print(f"Error at restric_val={val}: {e}")

        if algo_igd:
            summary[algo] = {
                'mean_igd': np.mean(algo_igd),
                'std_igd': np.std(algo_igd),
                'mean_hv': np.mean(algo_hv),
                'std_hv': np.std(algo_hv),
                'igd_list': algo_igd,
                'hv_list': algo_hv
            }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n{'Algorithm':<20} {'Mean IGD':<12} {'Std IGD':<12} {'Mean HV':<12} {'Std HV':<12}")
    print("-" * 68)
    for algo, stats in summary.items():
        print(f"{algo:<20} {stats['mean_igd']:<12.4f} {stats['std_igd']:<12.4f} {stats['mean_hv']:<12.4f} {stats['std_hv']:<12.4f}")

    return summary

def main():
    parser = argparse.ArgumentParser(description='Compare algorithm results on CA_Optimization')

    parser.add_argument('--optim_dir', type=str, default='./CA_Optimization',
                        help='Directory containing algorithm result folders')

    args = parser.parse_args()

    print("=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    print(f"Results dir: {args.optim_dir}")

    summary = run_comparison(optim_dir=args.optim_dir)

if __name__ == '__main__':
    main()
