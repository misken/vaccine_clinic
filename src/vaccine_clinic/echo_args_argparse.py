import sys
import argparse

def process_command_line(argv=None):
    """
    Parse command line arguments
    
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # If argv is empty, get the argument list from sys.argv.
    if argv is None:
        argv = sys.argv[1:]

    # Create the parser
    parser = argparse.ArgumentParser(prog='vaccine_clinic_model4',
                                     description='Run vaccine clinic simulation')

    # Add arguments
    parser.add_argument("--iat", help="patients per hour",
                        type=float)

    parser.add_argument("--greet", help="number of greeters",
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