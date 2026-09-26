from arch.config import *
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from arch.SAC_arch import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class MOO_Problem:

    def __init__(self, objectives, constraints, bounds):

        self.objectives = objectives
        self.constraints = constraints
        self.bounds = bounds
        self.obj_dim = len(objectives)
        self.constraint_dim = len(constraints)
        self.state_dim = len(bounds)

    def evaluate(self, x):

        x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        fs = np.array([f(x) for f in self.objectives])
        constraint_vals = np.array([c(x) for c in self.constraints])
        return fs, constraint_vals

class PymooMOO(ElementwiseProblem):
    def __init__(self, moo_problem: MOO_Problem):
        super().__init__(n_var=moo_problem.state_dim,
                         n_obj=moo_problem.obj_dim,
                         n_constr=moo_problem.constraint_dim,
                         xl=np.array([b[0] for b in moo_problem.bounds]),
                         xu=np.array([b[1] for b in moo_problem.bounds]))
        self.moo = moo_problem

    def _evaluate(self, x, out, *args, **kwargs):
        f_vals, c_vals = self.moo.evaluate(x)
        out["F"] = f_vals
        out["G"] = c_vals

class Simulator():
  def __init__(self, X: np.ndarray, y: np.ndarray, use_xgboost=True):

    from sklearn.preprocessing import MinMaxScaler
    if X.shape[0] != y.shape[0]:
        raise IndexError('X and y first dimension should be the same')
    self.model = self.__train_model(X, y, xgboost=use_xgboost)

    self.feature_dim = X.shape[1]
    self.x_scaler = MinMaxScaler()
    self.kpi_scaler = MinMaxScaler()
    train_x_arr = self.x_scaler.fit_transform(X)

    self.kpi_scaler.fit(y)
    self.X = X

  def __train_model(self, X, y, xgboost=True):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(random_state=42)
    model.fit(X, y)
    return model

  def __init_sample(self, inputs):
    return self.get_predict(inputs)

  def get_predict(self, inputs):
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    return self.model.predict(inputs)

  def sample(self, num=1):
    return np.squeeze(self.X[np.random.choice(self.X.shape[0], size=num)], axis=0).astype(np.float64)

def get_input_title():
  loc_paras = config_dic_wanhua['input_loc_paras']
  glo_paras = config_dic_wanhua['input_glo_paras']
  nod = config_dic_wanhua['NOD']
  inp_tits = []
  for i in range(nod):
    for j in range(len(loc_paras)):
      inp_tits.append("%s_%d"%(loc_paras[j], i+1))

  for j in range(len(glo_paras)):
    inp_tits.append("%s"%(glo_paras[j]))
  return inp_tits

def get_output_title():
  glo_paras = config_dic_wanhua['output_glo_paras']
  loc_paras_prefix = config_dic_wanhua['output_loc_paras']
  nod = config_dic_wanhua['NOD']
  out_tits = []
  for param in glo_paras:
    out_tits.append(param)
  for i in range(1, nod+1):
    for loc_prefix in loc_paras_prefix:
      out_tits.append(f'{loc_prefix}_{i}')
  return out_tits

def get_dataset_input(dataset):
  inp_tits = get_input_title()
  return dataset[inp_tits]

def get_dataset_output(dataset):
	out_tits = get_output_title()
	return dataset[out_tits]

def chlor_alkali_data(val_size: float, uniform=False):
  dataset = pd.read_csv('ChlorAlkali/wanhua.csv')
  input_data = get_dataset_input(dataset)
  output_data = get_dataset_output(dataset)
  def cal_daily(prod, den, con):

      step = config_dic_wanhua["frequency"]
      daily = prod*(den/1000)*(con/100)*24*(60/step)
      return daily
  def kpis_cal(predict_res, inputs):
      nod = config_dic_wanhua["NOD"]
      con = []
      curr_list = []
      for i in range(nod):
        con.append(inputs[:, (i+1)*7-1].reshape(-1, 1))
        curr_list.append(inputs[:, i*7].reshape(-1, 1))
      con = np.concatenate(con, axis=1)
      curr_list = np.concatenate(curr_list, axis=1)
      con = np.sum(con, axis=1)
      con /= nod
      leiji = predict_res[:, 0]
      den = predict_res[:, 1]

      prod = cal_daily(leiji, den, con)

      alp = config_dic_wanhua["KPI"]["alp"]
      nous = config_dic_wanhua["NOU"]
      curr_loc = []
      for i in range(nod):
        curr_loc.append((alp * nous[i] * 24 * curr_list[:, i]).reshape(-1, 1))
      curr_loc = np.concatenate(curr_loc, axis=1)
      curr_eff = prod*1000*100/(np.sum(curr_loc, axis=1)+1e-6)

      power = []
      for i in range(nod):
        I = curr_list[:, i]
        V = predict_res[:, 2+i]
        power.append((I*V).reshape(-1, 1))
      power = np.concatenate(power, axis=1)
      power = np.sum(power, axis=1)
      djdh = power*24/prod
      return np.concatenate([prod.reshape(-1, 1), curr_eff.reshape(-1, 1), djdh.reshape(-1, 1), predict_res[:, -8:]], axis=1)
  kpis = kpis_cal(output_data.values, input_data.values)

  train_x, val_x, train_y, val_y = train_test_split(input_data, kpis, test_size=val_size, random_state=42)
  if uniform:
    scaler = StandardScaler()
    train_x_arr = scaler.fit_transform(train_x)
    val_x_arr = scaler.transform(val_x)
    train_x = pd.DataFrame(train_x_arr, columns=train_x.columns, index=train_x.index)
    val_x = pd.DataFrame(val_x_arr, columns=val_x.columns, index=val_x.index)

  return train_x, train_y, val_x, val_y

def get_CA_data():

  y_cols = ['daily_prod','curr_eff','djdh']

  train_data = pd.read_csv("CA_train.csv")
  test_data = pd.read_csv("CA_test.csv")

  train_x = train_data.drop(columns=y_cols).values
  train_y = train_data[y_cols].values

  test_x = test_data.drop(columns=y_cols).values
  test_y = test_data[y_cols].values

  return train_x, train_y, test_x, test_y

def extract_res(y_min: np.ndarray, weights, top_k=50):
    if len(weights) != y_min.shape[1]:
        raise ValueError("weights shape should fit with y_min")
    w = np.array(weights, dtype=float)

    if np.any(w < 0):
        raise ValueError("All weights must be nonnegative.")

    s = w.sum()
    if s == 0:
        raise ValueError("At least one weight must be positive for normalization.")

    w = w / s

    scores = y_min.dot(weights)
    idx = np.argsort(scores)[:top_k]
    return idx

class TempSimu:
  def __init__(self, model, scaler, simu: Simulator):
    self.model = model
    self.scaler = scaler
    self.kpi_scaler = simu.kpi_scaler
    self.last_inputs = None
    self.last_fs = None

  def format_inputs(self, request):
    nod = config_dic_wanhua['NOD']
    loc_inp_tit = config_dic_wanhua['input_loc_paras']
    glo_inp_tit = config_dic_wanhua['input_glo_paras']
    inputs = []
    for ele in loc_inp_tit:
      for i in range(nod):
        key = "%s_%d"%(ele, i+1)
        try:
          inputs.append(float(request[key]))
        except:
          inputs.append(0)

    for ele in glo_inp_tit:
      key = "%s"%(ele)
      try:
        inputs.append(float(request[key]))
      except:
        inputs.append(0)
    return inputs
  def get_predict(self, inputs):
    if inputs.ndim == 2:
      inputs = inputs.squeeze()
    if self.last_inputs is not None and np.all(inputs == self.last_inputs):
        return self.last_fs
    else:
      self.last_inputs = inputs
    nod = config_dic_wanhua['NOD']
    loc_inp_tit = config_dic_wanhua['input_loc_paras']
    glo_inp_tit = config_dic_wanhua['input_glo_paras']

    nn_inputs = inputs
    nn_inputs = self.scaler.transform(nn_inputs.reshape(1, -1))
    format_inp = []
    for i in range(nod):
      format_inp.append(np.concatenate([nn_inputs[:, i*7:i*7+7], [nn_inputs[:, -2]], [nn_inputs[:, -1]]], axis=1))

    res = self.model.predict(format_inp, verbose=0)

    naoh_con_arr = [inputs[i*7+6] for i in range(nod)]
    con = sum(naoh_con_arr)/nod
    leiji = res[0][0]
    den = res[0][1]
    prod = cal_daily(leiji, den, con)
    curr_arr = [inputs[i*7] for i in range(nod)]
    curr_eff = cal_ce(prod, curr_arr)

    power = 0
    for i in range(nod):
      I = curr_arr[i]
      V = res[0][2+i]
      power += I*V
    djdh = power*24/prod

    res_dict = {}
    res_dict["daily_prod"] = str(round(prod,2))
    res_dict["current_eff"] = str(round(curr_eff,2))
    res_dict["power_per_prod"] = str(round(djdh,2))
    fs = np.array([[float(res_dict['daily_prod']), float(res_dict['current_eff']), float(res_dict['power_per_prod'])]])
    self.last_fs = fs
    return fs

import onnxruntime as ort

def run(initial_pop_size=100, max_steps=1000, max_size=20000, buffer_class_list=[BufferClass.FIFO], enable_rl=True, enable_random=True, GE_file='NSGAII', net_arch=64, res_save_dir='CA_Optimization'):
    import os
    print(f"PID: {os.getpid()}")
    train_x, train_y, test_x, test_y = get_CA_data()

    simulator = Simulator(train_x, train_y, use_xgboost=False)

    session = ort.InferenceSession('CA_model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def get_optimize_no_deviation(restric_val=86, GE_from_file=None, enable_rl=True, buffer_class=BufferClass.DPER, net_arch_hidden_dim=net_arch):
      def obj1(inputs):
          if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
          kpis_res: np.ndarray = session.run([output_name], {input_name: inputs})[0]

          kpis_res = kpis_res.transpose()
          kpis_res = simulator.kpi_scaler.transform(kpis_res)

          return -kpis_res[0, 0]

      def obj2(inputs):
          if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
          kpis_res = session.run([output_name], {input_name: inputs})[0]
          kpis_res = kpis_res.transpose()

          kpis_res[0, 1] = np.clip(kpis_res[0, 1], 0, 100)
          kpis_res = simulator.kpi_scaler.transform(kpis_res)

          return -kpis_res[0, 1]

      def obj3(inputs):
          if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
          kpis_res = session.run([output_name], {input_name: inputs})[0]
          kpis_res = kpis_res.transpose()

          kpis_res = simulator.kpi_scaler.transform(kpis_res)

          return kpis_res[0, 2]

      def current_constraint(inputs):
        values = [inputs[i*7] for i in range(nod)]
        return max(0, sum(values)-restric_val)

      def current_eff_constraint(inputs):
        kpis_res = simulator.get_predict(inputs)
        return max(0, kpis_res[0, 1]-100)

      params_range = []
      for i in range(train_x.shape[1]):
          low = np.min(train_x[:, i])
          high = np.max(train_x[:, i])
          params_range.append([low, high])

      nod = config_dic_wanhua['NOD']
      for i in range(nod):
          params_range[i*7][0] = restric_val/nod - 0.5
          params_range[i*7][1] = restric_val/nod + 0.5

      constraints_list = [current_constraint, current_eff_constraint]
      problem = MOO_Problem([obj1, obj2, obj3], constraints_list, bounds=params_range)

      if enable_rl:
        sac = SAC_Weighted_Arch(problem, net_arch_hidden_dim=net_arch_hidden_dim, penalty_coeff=10,
                            buffer_class=buffer_class, tensorboard_log='./tensorboard_logs/chlor_alkali/')
        path = 'sac_chlor_hidden32(state,'
        if buffer_class == BufferClass.DPER:
          path += 'DPER)'
        elif buffer_class == BufferClass.FIFO:
          path += 'FIFO)'
        elif buffer_class == BufferClass.PER:
          path += 'PER)'
        model_path = './CA_Model/' + path + '.zip'
        if os.path.exists(model_path):
          sac.load('./CA_Model/' + path)
        else:
          sac.learn(initial_population_size=50, steps_per_episode=100, batch_size=64, use_nsga2=False)
          sac.save('./CA_Model/' + path)

        X, O, C = sac.optimize(start_points_sample=initial_pop_size, n_gen=200, max_steps=max_steps,
                               GE_from_file=GE_from_file, max_size=max_size)

      else:
        X, O, C = random_optimize(problem=problem, start_points_sample=initial_pop_size, n_gen=200,
                                  max_steps=max_steps, GE_from_file=GE_from_file, max_size=max_size)

      return X, O, C

    def run_a_value(restric_val, GE_file='NSGAII', enable_rl=True, enable_random=True, res_save_dir='CA_Optimization'):
      GE_from_file = f'./CA_Optimization/{GE_file}/restric_val{restric_val}.txt'
      if enable_rl:
        for buffer_class in buffer_class_list:
          X, O, C = get_optimize_no_deviation(restric_val, GE_from_file=GE_from_file,
                                enable_rl=True, buffer_class=buffer_class)
          dir_path = f'{GE_file}+'
          dir_path += 'rl+'
          if buffer_class == BufferClass.DPER:
            dir_path += 'DPER'
          elif buffer_class == BufferClass.FIFO:
            dir_path += 'FIFO'
          elif buffer_class == BufferClass.PER:
            dir_path += 'PER'
          kpis = []
          for x in X:
            predict_res = simulator.get_predict(x)
            kpis.append(predict_res[0])
          os.makedirs(f'./{res_save_dir}/{dir_path}', exist_ok=True)
          with open(f'./{res_save_dir}/{dir_path}/restric_val{restric_val}.txt', 'w') as f:
            for x, fs, kpi in zip(X, O, kpis):
              f.write(f'X={x}\n')
              f.write(f'F={fs}\n')
              f.write(f'KPI={kpi}\n')
              f.write('-'*40)
              f.write('\n')

      if enable_random:
        dir_path = f'{GE_file}+'
        dir_path += 'random'
        X, O, C = get_optimize_no_deviation(restric_val, GE_from_file=GE_from_file,
                                enable_rl=False, buffer_class=BufferClass.FIFO)
        kpis = []
        for x in X:
          predict_res = simulator.get_predict(x)
          kpis.append(predict_res[0])
        os.makedirs(f'./{res_save_dir}/{dir_path}', exist_ok=True)
        with open(f'./{res_save_dir}/{dir_path}/restric_val{restric_val}.txt', 'w') as f:
          for x, fs, kpi in zip(X, O, kpis):
            f.write(f'X={x}\n')
            f.write(f'F={fs}\n')
            f.write(f'KPI={kpi}\n')
            f.write('-'*40)
            f.write('\n')

    restric_vals = range(86, 120)
    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(run_a_value, val, GE_file, enable_rl, enable_random, res_save_dir) for val in restric_vals]
      for f in futures:
        try:
          f.result()
        except Exception as e:
          print(f'Error occur: {e}')

if __name__ == '__main__':
  buffer_class_list = [BufferClass.FIFO, BufferClass.PER]
  enable_rl = True
  enable_random = True

  GE_files = ['C3M', 'cDPEA', 'CMDEIPCM','IDBEA', 'MOEADD','NSGAII', 'SSDE','ToP']
  for GE in GE_files:
    run(
        initial_pop_size=100,
        max_steps=1000,
        max_size=5000,
        buffer_class_list=buffer_class_list,
        enable_rl=enable_rl,
        enable_random=enable_random,
        GE_file=GE,
        net_arch=32,
        res_save_dir='CA_Optimization_with_SPEA2'
    )

