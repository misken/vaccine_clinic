# get_option_values.py
import sys

def main():

    input_params = {'mean_interarrival_time': 0.0,
                    'num_greeters': 0,
                    'num_reg_staff': 0,
                    'num_vaccinators': 0,
                    'num_schedulers': 0}


    for i, arg in enumerate(sys.argv):
        if arg.startswith('--') and i % 2 > 0:
            if sys.argv[i] == '--iat':
                input_params['mean_interarrival_time'] = sys.argv[i + 1]
            elif sys.argv[i] == '--greet':
                input_params['num_greeters'] = sys.argv[i + 1]
            elif sys.argv[i] == '--reg':
                input_params['num_reg_staff'] = sys.argv[i + 1]
            elif sys.argv[i] == '--vacc':
                input_params['num_vaccinators'] = sys.argv[i + 1]
            elif sys.argv[i] == '--sched':
                input_params['num_schedulers'] = sys.argv[i + 1]
            else:
                print(f"Unrecognized argument: {sys.argv[i]}")

    print(input_params)

if __name__ == '__main__':
   main()