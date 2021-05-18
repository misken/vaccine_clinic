from vaccine_clinic_model4 import simulate

args = {'patient_arrival_rate': 180,
        'num_greeters': 4,
        'num_reg_staff': 4,
        'num_vaccinators': 15,
        'num_schedulers': 4,
        'scenario': 'base',
        'pct_need_second_dose': 0.50,
        'temp_check_time_mean': 0.25,
        'temp_check_time_sd': 0.05,
        'reg_time_mean': 1.0,
        'vaccinate_time_mean': 4.0,
        'vaccinate_time_sd': 0.5,
        'sched_time_mean': 1.00,
        'sched_time_sd': 0.10,
        'obs_time': 15.0,
        'post_obs_time_mean': 1.0,
        'stoptime': 600,
        'num_reps': 1,
        'seed': 4470,
        'output_path': 'output',
        'quiet': True}

num_reps = args['num_reps']

for i in range(1, num_reps + 1):
    simulate(args, i)

config_fn = f"{args['scenario']}.cfg"

with open(config_fn, "w") as config_file:
    for key, value in args.items():
        if key != 'config':
            config_file.write(f"--{key} {value}\n")
