from pathlib import Path

from vaccine_clinic_model4 import simulate, process_sim_output

args = {'patient_arrival_rate': 180,
        'num_greeters': 3,
        'num_reg_staff': 3,
        'num_vaccinators': 12,
        'num_schedulers': 3,
        'scenario': 'base_g3r3v12s3',
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
        'num_reps': 15,
        'seed': 4470,
        'output_path': 'output',
        'quiet': True}

num_reps = args['num_reps']
scenario = args['scenario']

if len(args['output_path']) > 0:
    output_dir = Path.cwd() / args['output_path']
else:
    output_dir = Path.cwd()

for i in range(1, num_reps + 1):
    simulate(args, i)

# Consolidate the patient logs
process_sim_output(output_dir, scenario)

# Create a config file based on the inputs above

config_fn = f"input/{args['scenario']}.cfg"

with open(config_fn, "w") as config_file:
    for key, value in args.items():
        if key != 'config':
            if key != 'quiet':
                config_file.write(f"--{key} {value}\n")
            else:
                if value:
                    config_file.write(f"--{key}\n")


