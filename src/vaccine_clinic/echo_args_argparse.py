import sys
import argparse

def process_command_line():
    """
    Parse command line arguments
    
    Return a Namespace representing the argument list.
    """


    # Create the parser
    parser = argparse.ArgumentParser(prog='vaccine_clinic_model4',
                                     description='Run vaccine clinic simulation')

    # Add arguments
    parser.add_argument("-i", "--iat", help="patients per hour",
                        type=float)

    parser.add_argument("-g", "--greet", help="number of greeters",
                        type=int)

    parser.add_argument("--reg", help="number of registration staff",
                        type=int)

    parser.add_argument("--vacc", help="number of vaccinators",
                        type=int)

    parser.add_argument("--sched", help="number of schedulers",
                        type=int)

    # do the parsing
    args = parser.parse_args()
    return args


def main():

    args = process_command_line()
    print("args.iat: ", args.iat)
    print("args: ", args)

    print("vars(args):", vars(args))


if __name__ == '__main__':
    main()