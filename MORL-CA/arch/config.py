config_dic_wanhua = {

	"input_glo_paras":["glo_temp", "temp_naoh"],
	"input_loc_paras":["current","temp","hcl","p_flow","n_flow","nacl_den","naoh_con"],
	"output_glo_paras":["prod_grow_per","naoh_den"],
	"output_loc_paras": ["vot"],
	"model_path":"cais_model/new_wanhua.h5",

	"predict_model":"NN",

	"NOD":8,
	"NOU":[164,164,164,164,164,164,164,164],

	"frequency": 5,
	"static_file_path": "static/data/wanhua2015",

	"frac": 0.8,
	"learning_rate": 0.001,
	"train_epoch":200,
	"lr_decay": 1e-5,
	"loss_component":[1, 0.1, 0.05],

	"incremental":{
		"warm_up_epoch":100,
		"joint_opt_epoch":100,
		"warm_up_lr":0.001,
		"joint_opt_lr":0.0001,
		"old_sampling_rate":0.1,
		"shared_paras":["current","temp"],
		"new_paras":[],
		"threshold_vot":5,
		"threshold_density":1,
		"threshold_quatity":0.1,
		"threshold_percent":0.05
	},

	"surface_resolution":20,
	"surface_delta": 0.2,
	"surface_para_selection": 2,

	"fixed_temp": 90,
	"fixed_naoh_con": 32,
	"ce_valid_range": [90, 99],

	"DB":{"db_name": "wanhua2015", "db_user":"root", "db_password":"123456", "db_char_len":64},
	"batch_size_db":2000,

	"KPI":{
		"window":864,
		"KT":0.0081,
		"KC":0.0199,
		"lam":5.5,
		"alp":1.492,
		"flow_ratio":0.75,
		"anode_eff_alp": 0.0373,
		"current_leak_ratio":0.3,
		"modify_vot_alp": 2.42,
		"base_temperature": 90,
		"base_density": 32,
		"current_lam_ratio": 2.7,
		"kpi_glo":["daily_prod","current_eff","power_per_prod","prod_grow_per","soc", "power"],
		"kpi_loc":["anode_curr_eff","unit_vot","modify_unit_vot","unit_curr","modify_unit_curr","modify2_unit_curr"]
	},

	"OPT":{"soc":[86,120], "prod":[500,650]},
	"opt_iter":200,
	"delta":1,
	"delta_prod":20,
	"lower_bound": 0.1,
	"upper_bound": 0.9,

	"valid_range": {
		"current_eff": [90,97],
		"power_per_prod": [2000, 2400],
		"modify_unit_vot": [2.4, 3.5],
		"temp_naoh": [80, 88],
		"naoh_con": [31, 32.5],
		"temp_naoh": [60, 100],
		"temp": [82,88],
		"nacl_den": [205, 215],
		"hcl":[0,700],
		"p_flow":[0,45],
		"n_flow":[0,55],
		"vot": [2.4, 3.5],
		"current": [12,15],
		"prod_grow_per": [0,10],
		"daily_prod":[480,650]
	},

	"max_range":{
		"temp": [60,100],
		"nacl_den": [205, 215],
		"hcl": [0,700],
		"p_flow":[0,45],
		"n_flow":[0,55],
		"vot": [2.4, 3.5],
		"current": [0,15.75],
		"prod_grow_per": [0, 2000],
		"temp_naoh": [75, 85],
		"naoh_con": [31, 32.5],
		"current_eff": [90,97],
		"daily_prod":[500,650],
		"power_per_prod": [1500, 2500]
	},

	"kpi_g1":["current","vot","p_flow","hcl","temp","acid_input","acid_output","n_flow","nacl_den"],
	"kpi_g2":["O_in_Cl","naoh_prod","naoh_con","naoh_den","temp_naoh"]

}

config_dic_binhua = {

	"input_glo_paras":["naoh_con","h2o_vapor", "Acid","h2o","GPD","pos_liq_level","neg_liq_level","H2","Cl2","nacl_temp"],
	"input_loc_paras":["current","temp","nacl_flow","reflux_naoh","LPD","LP_pos","LP_neg","out_nacl_PH","in_nacl_PH"],
	"output_glo_paras":["prod_grow_per","naoh_den"],
	"output_loc_paras": ["vot"],
	"model_path":"cais_model/model_current.h5",

	"NOD":6,
	"NOU":195,
	"frequency": 20,

	"frac": 0.8,
	"train_epoch":500,
	"learning_rate": 0.001,
	"lr_decay": 1e-5,
	"static_file_path": "static/data/binhua2020",
	"loss_component":[1, 0, 0.05],
	"incremental":{
		"warm_up_epoch":100,
		"joint_opt_epoch":100,
		"warm_up_lr":0.001,
		"joint_opt_lr":0.0001,
		"old_sampling_rate":0.1,
		"shared_paras":["current","temp"],
		"new_paras":[]
	},

	"DB":{
		"db_name": "binhua2020",
		"db_user":"root",
		"db_password":"123456",
		"db_char_len":64
	},

	"batch_size_db":2000,

	"surface_delta": 0.2,

	"surface_para_selection": 2,
	"fixed_temp": 90,
	"fixed_naoh_con": 32,
	"ce_valid_range": [90, 99],

	"delta":0.2,
	"opt_iter":300,
	"lower_bound": 0.1,
	"upper_bound": 0.9,
	"OPT":{"soc":[86,120], "prod":[500,650]},

	"predict_model":"NN",
	"surface_resolution":20,

	"KPI":{
		"window":216,
		"KT":0.0081,
		"KC":0.0199,
		"lam":5.5,
		"alp":1.492,
		"flow_ratio":0.75,
		"anode_eff_alp": 0.0373,
		"current_leak_ratio":0.3,
		"modify_vot_alp": 2.42,
		"base_temperature": 90,
		"base_density": 32,
		"current_lam_ratio": 2.7,
		"kpi_glo":["daily_prod","current_eff","power_per_prod","prod_grow_per","soc", "power"],
		"kpi_loc":["unit_vot"],
		"default_naoh_den": 1300
	},

	"valid_range": {
		"current_eff": [90,97],
		"power_per_prod": [2000, 2400],
		"modify_unit_vot": [2.4, 3.5],
		"temp_naoh": [80, 88],
		"naoh_con": [31, 32.5],
		"temp_naoh": [60, 100],
		"temp": [82,88],
		"nacl_den": [205, 215],
		"hcl":[0,700], "p_flow":[0,45],
		"n_flow":[0,55],
		"vot": [2.4, 3.5],
		"current": [12,15],
		"naoh_prod": [0,6],
	},

	"max_range":{
		"temp": [60,100],
		"nacl_den": [205, 215],
		"hcl": [0,700],
		"p_flow":[0,45],
		"n_flow":[0,55],
		"vot": [2.4, 3.5],
		"current": [0,15.75],
		"prod_grow_per": [0, 2000],
		"temp_naoh": [75, 85],
		"naoh_con": [31, 32.5],
		"current_eff": [90,97],
		"power_per_prod": [2000, 2400]
	},

	"kpi_g1":["current","vot","p_flow","hcl","temp","acid_input","acid_output","n_flow","nacl_den"],
	"kpi_g2":["O_in_Cl","naoh_prod","naoh_con","naoh_den"]

}
