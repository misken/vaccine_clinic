#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng
import simpy
from pathlib import Path


# ### Possible improvements
# 
# Now that we have a rough first model working, let's think about some possible improvements.
# There are many as we've taken numerous shortcuts to get this model working.
# 
# * Specifying the global sim inputs through configuration files
# * Getting rid of hard coded processing time distributions
# * Having ability to choose between pure random walk-in arrivals and scheduled arrivals
# * Multiple replications and/or steady state analysis
# * Detailed logging
# * Statistical summaries based on timestamp data [roughly done]
# * CLI for running
# * Animation
# 

# ## Model 3: The vaccine clinic model - version 0.01
# 
# Here's the basic vaccination process we'll model.
# 
# Arrival --> Temperature check --> Registration --> Vaccination --> Sched dose 2 (if needed) --> Wait 15 min --> Exit

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
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--patient_arrival_rate", default=150, help="patients per hour",
                        type=float)

    parser.add_argument("--num_greeters", default=2, help="number of greeters",
                        type=int)

    parser.add_argument("--num_reg_staff", default=2, help="number of registration staff",
                        type=int)

    parser.add_argument("--num_vaccinators", default=15, help="number of vaccinators",
                        type=int)

    parser.add_argument("--num_schedulers", default=2, help="number of schedulers",
                        type=int)

    parser.add_argument("--pct_need_second_dose", default=0.5,
                        help="percent of patients needing 2nd dose (default = 0.5)",
                        type=float)

    parser.add_argument("--obs_time", default=15,
                        help="Time (minutes) patient waits post-vaccination in observation area (default = 15)",
                        type=float)

    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Appended to output filenames."
    )

    parser.add_argument("--stoptime", default=600, help="time that simulation stops (default = 600)",
                        type=float)

    parser.add_argument("--num_reps", default=1, help="number of simulation replications (default = 1)",
                        type=float)

    parser.add_argument("--seed", default=3, help="random number generator seed (default = 3)",
                        type=float)

    parser.add_argument(
        "--output_path", type=str, default="", help="location for output file writing")

    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False")

    # do the parsing
    args = parser.parse_args()
    return args


class PatientFlow(object):
    def __init__(self, mean_interarrival_time, pct_need_second_dose):
        self.mean_interarrival_time = mean_interarrival_time
        self.pct_need_second_dose = pct_need_second_dose




class VaccineClinic(object):
    def __init__(self, env, num_greeters, num_reg_staff, num_vaccinators, num_schedulers):
        # Simulation environment
        self.env = env

        # Create list to hold timestamps dictionaries (one per patient)
        self.timestamps_list = []
        # Create lists to hold occupancy tuples (time, occ)
        self.postvac_occupancy_list = [(0.0, 0.0)]
        self.vac_occupancy_list = [(0.0, 0.0)]

        # Create resources
        self.greeter = simpy.Resource(env, num_greeters)
        self.reg_staff = simpy.Resource(env, num_reg_staff)
        self.vaccinator = simpy.Resource(env, num_vaccinators)
        self.scheduler = simpy.Resource(env, num_schedulers)

    # Create process methods - hard coding processing time distributions for now
    # The patient argument is just a unique integer number
    def temperature_check(self, rg):
        yield self.env.timeout(rg.normal(0.25, 0.05))

    def registration(self, rg):
        yield self.env.timeout(rg.exponential(1.0))

    def vaccinate(self, rg):
        yield self.env.timeout(rg.normal(4.0, 0.5))

    def schedule_dose_2(self, rg):
        yield self.env.timeout(rg.normal(1.0, 0.10))

    # We assume all patients wait at least 15 minutes post-vaccination
    # Some will choose to wait longer. This is the time beyond 15 minutes
    # that patients wait.
    def wait_gt_15(self, rg):
        yield self.env.timeout(rg.exponential(0.5))


#

def get_vaccinated(env, patient, clinic, patient_flow, rg, quiet):
    """Defines the sequence of steps traversed by patients.

       Also capture a bunch of timestamps to make it easy to compute various system
       performance measures such as patient waiting times, queue sizes and resource utilization.
    """
    # Patient arrives to clinic - note the arrival time
    arrival_ts = env.now

    # Request a greeter for temperature check
    # By using request() in a context manager, we'll automatically release the resource when done
    with clinic.greeter.request() as request:
        yield request
        # Now that we have a greeter, check temperature. Note time.
        got_greeter_ts = env.now
        yield env.process(clinic.temperature_check(rg))
        release_greeter_ts = env.now

    # Request reg staff to get registered
    with clinic.reg_staff.request() as request:
        yield request
        got_reg_ts = env.now
        yield env.process(clinic.registration(rg))
        release_reg_ts = env.now

    # Request clinical staff to get vaccinated
    with clinic.vaccinator.request() as request:
        yield request
        got_vaccinator_ts = env.now
        # Update vac occupancy - increment by 1
        prev_occ = clinic.vac_occupancy_list[-1][1]
        new_occ = (env.now, prev_occ + 1)
        clinic.vac_occupancy_list.append(new_occ)
        yield env.process(clinic.vaccinate(rg))
        release_vaccinator_ts = env.now
        # Update vac occupancy - decrement by 1 - more compact code
        clinic.vac_occupancy_list.append((env.now, clinic.vac_occupancy_list[-1][1] - 1))

        # Update postvac occupancy - increment by 1
        clinic.postvac_occupancy_list.append((env.now, clinic.postvac_occupancy_list[-1][1] + 1))

    # Request scheduler to schedule second dose if needed
    if rg.random() < patient_flow.pct_need_second_dose:
        with clinic.scheduler.request() as request:
            yield request
            got_scheduler_ts = env.now
            yield env.process(clinic.schedule_dose_2(rg))
            release_scheduler_ts = env.now
    else:
        got_scheduler_ts = pd.NA
        release_scheduler_ts = pd.NA

    # Wait at least 15 minutes from time we finished getting vaccinated 
    post_vac_time = env.now - release_vaccinator_ts
    if post_vac_time < 15:
        # Wait until 15 total minutes post vac
        yield env.timeout(15 - post_vac_time)
        # Wait random amount beyond 15 minutes
        yield env.process(clinic.wait_gt_15(rg))

        # Update postvac occupancy - decrement by 1
        clinic.postvac_occupancy_list.append((env.now, clinic.postvac_occupancy_list[-1][1] - 1))

    exit_system_ts = env.now
    if not quiet:
        print(f"Patient {patient} exited at time {env.now}")

    # Create dictionary of timestamps
    timestamps = {'patient_id': patient,
                  'arrival_ts': arrival_ts,
                  'got_greeter_ts': got_greeter_ts,
                  'release_greeter_ts': release_greeter_ts,
                  'got_reg_ts': got_reg_ts,
                  'release_reg_ts': release_reg_ts,
                  'got_vaccinator_ts': got_vaccinator_ts,
                  'release_vaccinator_ts': release_vaccinator_ts,
                  'got_scheduler_ts': got_scheduler_ts,
                  'release_scheduler_ts': release_scheduler_ts,
                  'exit_system_ts': exit_system_ts}

    clinic.timestamps_list.append(timestamps)


def run_clinic(env, clinic, patient_flow, rg,
               stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, quiet=False):
    """
    Run the clinic for a specified amount of time.

    Parameters
    ----------
    env
    clinic
    patient_flow
    rg
    stoptime
    max_arrivals
    quiet

    Returns
    -------

    """

    # Create a counter to keep track of number of patients generated and to serve as unique patient id
    patient = 0

    # Loop for generating patients
    while env.now < stoptime and patient < max_arrivals:
        # Generate next interarrival time (this will be more complicated later)
        iat = rg.exponential(patient_flow.mean_interarrival_time)

        # This process will now yield to a 'timeout' event. This process will resume after iat time units.
        yield env.timeout(iat)

        # New patient generated = update counter of patients
        patient += 1

        if not quiet:
            print(f"Patient {patient} created at time {env.now}")

        env.process(get_vaccinated(env, patient, clinic, patient_flow, rg, quiet))

    print(f"{patient} patients processed.")


def compute_durations(timestamp_df):
    """Compute time durations of interest from timestamps dataframe and append new cols to dataframe"""

    timestamp_df['wait_for_greeter'] = timestamp_df.loc[:, 'got_greeter_ts'] - timestamp_df.loc[:, 'arrival_ts']
    timestamp_df['wait_for_reg'] = timestamp_df.loc[:, 'got_reg_ts'] - timestamp_df.loc[:, 'release_greeter_ts']
    timestamp_df['wait_for_vaccinator'] = timestamp_df.loc[:, 'got_vaccinator_ts'] - timestamp_df.loc[:,
                                                                                     'release_reg_ts']
    timestamp_df['vaccination_time'] = timestamp_df.loc[:, 'release_vaccinator_ts'] - timestamp_df.loc[:,
                                                                                      'got_vaccinator_ts']
    timestamp_df['wait_for_scheduler'] = timestamp_df.loc[:, 'got_scheduler_ts'] - timestamp_df.loc[:,
                                                                                   'release_vaccinator_ts']
    timestamp_df['post_vacc_time'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:,
                                                                             'release_vaccinator_ts']
    timestamp_df['time_in_system'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'arrival_ts']

    return timestamp_df


def simulate(arg_dict, rep_num):

    patient_arrival_rate = arg_dict['patient_arrival_rate']
    mean_interarrival_time = 1.0 / (patient_arrival_rate / 60.0)
    pct_need_second_dose = arg_dict['pct_need_second_dose']

    # Create a PatientFlow object to store patient flow related parameters
    patient_flow = PatientFlow(mean_interarrival_time, pct_need_second_dose)

    # Create a random number generator
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_greeters = arg_dict['num_greeters']
    num_reg_staff = arg_dict['num_reg_staff']
    num_vaccinators = arg_dict['num_vaccinators']
    num_schedulers = arg_dict['num_schedulers']

    # Other parameters
    stoptime = arg_dict['stoptime']  # No more arrivals after this time
    quiet = arg_dict['quiet']
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()
    # Create a clinic to simulate
    clinic = VaccineClinic(env, num_greeters, num_reg_staff, num_vaccinators, num_schedulers)

    env.process(
        run_clinic(env, clinic, patient_flow, rg, stoptime=stoptime, quiet=quiet))
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd() / 'output'
    print(output_dir)

    # Create paths for the output logs
    clinic_patient_log_path = output_dir / f'clinic_patient_log_{scenario}_{rep_num}.csv'
    vac_occupancy_df_path = output_dir / f'vac_occupancy_{scenario}_{rep_num}.csv'
    postvac_occupancy_df_path = output_dir / f'postvac_occupancy_{scenario}_{rep_num}.csv'

    # Create patient log dataframe and add scenario and rep number cols
    clinic_patient_log_df = pd.DataFrame(clinic.timestamps_list)
    clinic_patient_log_df['scenario'] = scenario
    clinic_patient_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(clinic_patient_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols-2)])
    clinic_patient_log_df = clinic_patient_log_df.iloc[:, new_col_order]

    # Compute durations of interest for patient log
    clinic_patient_log_df = compute_durations(clinic_patient_log_df)

    # Create occupancy log dataframes and add scenario and rep number cols
    vac_occupancy_df = pd.DataFrame(clinic.vac_occupancy_list, columns=['ts', 'occ'])
    vac_occupancy_df['scenario'] = scenario
    vac_occupancy_df['rep_num'] = scenario
    num_cols = len(vac_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    vac_occupancy_df = vac_occupancy_df.iloc[:, new_col_order]

    postvac_occupancy_df = pd.DataFrame(clinic.postvac_occupancy_list, columns=['ts', 'occ'])
    postvac_occupancy_df['scenario'] = scenario
    postvac_occupancy_df['rep_num'] = scenario
    num_cols = len(postvac_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    postvac_occupancy_df = vac_occupancy_df.iloc[:, new_col_order]

    # Export logs to csv
    clinic_patient_log_df.to_csv(clinic_patient_log_path, index=False)
    vac_occupancy_df.to_csv(vac_occupancy_df_path, index=False)
    postvac_occupancy_df.to_csv(postvac_occupancy_df_path, index=False)

    # Summary stats - keep or move?
    summary_stats_path = output_dir / f'summary_stats_{scenario}_{rep_num}.csv'
    summary_stats_df = clinic_patient_log_df.loc[:, ['wait_for_vaccinator', 'time_in_system']].describe()
    summary_stats_df.to_csv(summary_stats_path, index=True)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation ended at time {end_time}")


def main():

    args = process_command_line()
    print(args)

    num_reps = args.num_reps
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)


if __name__ == '__main__':
    main()


