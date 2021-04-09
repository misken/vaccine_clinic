#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

import simpy


# ### Possible improvements
# 
# Now that we have a rough first model working, let's think about some possible improvements. There are many as we've taken numerous shortcuts to get this model working.
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
#     Arrival --> Temperature check --> Registration --> Vaccination --> Schedule dose 2 (if needed) --> Wait 15 min --> Exit

class VaccineClinic(object):
    def __init__(self, env, num_greeters, num_reg_staff, num_vaccinators, num_schedulers, rg):
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
    def temperature_check(self, patient, rg):
        yield self.env.timeout(rg.normal(0.25, 0.05))

    def registration(self, patient, rg):
        yield self.env.timeout(rg.exponential(1.0))

    def vaccinate(self, patient, rg):
        yield self.env.timeout(rg.normal(4.0, 0.5))
        
    def schedule_dose_2(self, patient, rg):
        yield self.env.timeout(rg.normal(1.0, 0.25))
    
    # We assume all patients wait at least 15 minutes post-vaccination
    # Some will choose to wait longer. This is the time beyond 15 minutes
    # that patients wait.
    def wait_gt_15(self, patient, rg):
        yield self.env.timeout(rg.exponential(0.5))


# Now create general function to define the sequence of steps traversed by patients. We'll also capture a bunch of timestamps to make it easy to compute various system performance measures such as patient waiting times, queue sizes and resource utilization.

def get_vaccinated(env, patient, clinic, pct_first_dose, rg):
    # Patient arrives to clinic - note the arrival time
    arrival_ts = env.now

    # Request a greeter for temperature check
    # By using request() in a context manager, we'll automatically release the resource when done
    with clinic.greeter.request() as request:
        yield request
        # Now that we have a greeter, check temperature. Note time.
        got_greeter_ts = env.now
        yield env.process(clinic.temperature_check(patient, rg))
        release_greeter_ts = env.now

    # Request reg staff to get registered
    with clinic.reg_staff.request() as request:
        yield request
        got_reg_ts = env.now
        yield env.process(clinic.registration(patient, rg))
        release_reg_ts = env.now
        
    # Request clinical staff to get vaccinated
    with clinic.vaccinator.request() as request:
        yield request
        got_vaccinator_ts = env.now
        # Update vac occupancy - increment by 1
        prev_occ = clinic.vac_occupancy_list[-1][1]
        new_occ = (env.now, prev_occ + 1)
        clinic.vac_occupancy_list.append(new_occ)
        yield env.process(clinic.vaccinate(patient, rg))
        release_vaccinator_ts = env.now
        # Update vac occupancy - decrement by 1 - more compact code
        clinic.vac_occupancy_list.append((env.now, clinic.vac_occupancy_list[-1][1] - 1))
        
        # Update postvac occupancy - increment by 1
        clinic.postvac_occupancy_list.append((env.now, clinic.postvac_occupancy_list[-1][1] + 1))

    # Request scheduler to schedule second dose if needed
    if rg.random() < pct_first_dose:
        with clinic.scheduler.request() as request:
            yield request
            got_scheduler_ts = env.now
            yield env.process(clinic.schedule_dose_2(patient, rg))
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
        yield env.process(clinic.wait_gt_15(patient, rg))
        exit_system_ts = env.now
        
        # Update postvac occupancy - decrement by 1
        clinic.postvac_occupancy_list.append((env.now, clinic.postvac_occupancy_list[-1][1] - 1))
    
    exit_system_ts = env.now    
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


# Now create a function that runs the clinic for a specified number of hours.
# 
# TODO: Dealing with the hours of operation and making sure clinic cleared at end of day. Create clinic level dict of stats such as number of patients vaccinated and end of day timestamp.


def run_clinic(env, clinic, mean_interarrival_time, pct_first_dose, rg, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity):
      
    # Create a counter to keep track of number of patients generated and to serve as unique patient id
    patient = 0

    # Loop for generating patients
    while env.now < stoptime and patient < max_arrivals:

        # Generate next interarrival time (this will be more complicated later)
        iat = rg.exponential(mean_interarrival_time)
        
        # This process will now yield to a 'timeout' event. This process will resume after iat time units.
        yield env.timeout(iat)

        # New patient generated = update counter of patients
        patient += 1
        
        print(f"Patient {patient} created at time {env.now}")

        env.process(get_vaccinated(env, patient, clinic, pct_first_dose, rg))
        


def main():
    
    # For now we are hard coding the patient arrival rate (patients per hour)
    patients_per_hour = 180
    mean_interarrival_time = 1.0 / (patients_per_hour / 60.0)
    pct_first_dose = 0.50
    
    # Create a random number generator
    rg = default_rng(seed=4470)
    
    # For now we are going to hard code in the resource capacity levels 
    num_greeters = 2
    num_reg_staff = 4
    num_vaccinators = 15
    num_schedulers = 4
    
    # Hours of operation
    stoptime = 600 # No more arrivals after this time
    
    
    # Run the simulation
    env = simpy.Environment()
    # Create a clinic to simulate
    clinic = VaccineClinic(env, num_greeters, num_reg_staff, num_vaccinators, num_schedulers, rg)
    
    env.process(run_clinic(env, clinic, mean_interarrival_time, pct_first_dose, rg, stoptime=stoptime))
    env.run()
    
    # Output log files
    clinic_patient_log_df = pd.DataFrame(clinic.timestamps_list)
    clinic_patient_log_df.to_csv('./output/clinic_patient_log_df.csv', index=False)
    
    vac_occupancy_df = pd.DataFrame(clinic.vac_occupancy_list, columns=['ts', 'occ'])
    vac_occupancy_df.to_csv('./output/vac_occupancy_df.csv', index=False)
    
    postvac_occupancy_df = pd.DataFrame(clinic.postvac_occupancy_list, columns=['ts', 'occ'])
    postvac_occupancy_df.to_csv('./output/postvac_occupancy_df.csv', index=False)
    
    # Note simulation end time
    end_time = env.now
    print(f"Simulation ended at time {end_time}")
    return (end_time)

# Call main() to run a simulation
clinic_end_time = main()


# ## Redesign for easier use
# 
# ### Hard coded input parameters
# 
# The current version of the model has many hard coded input parameters, including:
# 
# * patient arrival rate,
# * percent of patients requiring 2nd dose,
# * capacity of the various resources (e.g. vaccinators),
# * processing time distributions for the various stages of the vaccination process,
# * length of time patient needs to remain in observation after being vaccinated (i.e. the 15 minutes),
# * length of time to run the simulation model.
# 
# This makes it cumbersome to try out different scenarios relating to patient volume and resource capacity.
# 
# ### Better post-processing
# 
# We should make it easy to specify that some post-processing should just happen automatically. In general, we want better control over post-processing and think through how we want to create and store output files.
# 
# ### Moving code from Jupyter notebook to a Python script
# 
# As our code grows, working inside a Jupyter notebook becomes less practical. It's fine for testing out little code snippets, but the production code should be in a `.py` file and we should use an IDE for further development. 
# 
# ### Adding a command line interface
# 
# By moving the code to a `.py` file, it will also be easier to add and use a simple command line interface. This will be useful for automating the process of running numerous scenarios.
# 

# In[ ]:




