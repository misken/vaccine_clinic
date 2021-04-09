from vaccine_clinic_model4 import simulate

args = {'patient_arrival_rate': 180,
        'num_greeters': 4,
        'num_reg_staff': 4,
        'num_vaccinators': 15,
        'num_schedulers': 4,
        'scenario': 'func_test',
        'pct_need_second_dose': 0.50,
        'stoptime': 600,
        'output_path': 'output',
        'quiet': True}

simulate(args)