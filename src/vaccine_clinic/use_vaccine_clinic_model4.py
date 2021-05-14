from vaccine_clinic_model4 import simulate

args = {'patient_arrival_rate': 180,
        'num_greeters': 4,
        'num_reg_staff': 4,
        'num_vaccinators': 15,
        'num_schedulers': 4,
        'scenario': 'base',
        'pct_need_second_dose': 0.50,
        'stoptime': 600,
        'num_reps': 5,
        'seed': 4470,
        'output_path': 'output',
        'quiet': True}

num_reps = args['num_reps']
for i in range(1, num_reps + 1):
    simulate(args, i)
