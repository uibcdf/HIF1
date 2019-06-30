import warnings
warnings.filterwarnings("ignore")

import os
import sys
import shutil
from time import time as realtime
from pytools_uibcdf.Time import formatted_elapsed_time, formatted_local_time
import molmodmt as m3t
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from mdtraj.reporters import HDF5Reporter
from openmmtools.integrators import LangevinIntegrator

#### SLURM and environment variables

slurm_hostname = os.environ['HOSTNAME']
slurm_jobname = os.environ['SLURM_JOB_NAME']
slurm_jobid = os.environ['SLURM_JOB_ID']
work_dir = os.getcwd()

#### Files Produced

## In current dir

log_file = 'logfile.txt'

## In temporal dir

tmp_dir ="/TempDATA/tmp_"+slurm_jobid
os.mkdir(tmp_dir)

trajectory_file = os.path.join(tmp_dir, "trajectory.h5")
checkpoint_file = os.path.join(tmp_dir, "checkpoint.chk")
final_state_file = os.path.join(tmp_dir, "final_state.xml")
final_checkpoint_file = os.path.join(tmp_dir, "final_checkpoint.chk")
final_pdb_file = os.path.join(tmp_dir, "final_positions.pdb")

#### Log

log = open(log_file,'w')
#log = sys.stdout

start_realtime = realtime()
log.write("\n")
log.write("Start: "+formatted_local_time()+"\n")
log.write("Hostname: "+slurm_hostname+"\n")
log.write("Jobname: "+slurm_jobname+"\n")
log.write("JobID: "+slurm_jobid+"\n")
log.write("\n")

#### Loading PDB

pdb = m3t.convert('system_equilibrated_NPT.pdb', 'openmm.PDBFile')

#### System

topology = m3t.convert(pdb, 'openmm.Topology')
forcefield = app.ForceField('amber99sbildn.xml','tip3p.xml')
system = forcefield.createSystem(topology,
                                 nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.2*unit.nanometers,
                                 constraints=app.HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

#### Thermodynamic State

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 300.0*unit.kelvin
pressure    = 1.0*unit.atmosphere

#### Integrator

friction   = 1.0/unit.picosecond
step_size  = 2.0*unit.femtoseconds
integrator = LangevinIntegrator(temperature, friction, step_size)
integrator.setConstraintTolerance(0.00001)

#### Barostat

barostat_interval = 25
barostat = mm.MonteCarloBarostat(pressure, temperature, barostat_interval)
system.addForce(barostat)

#### Platform

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

#### Simulation

simulation = app.Simulation(topology, system, integrator, platform, properties)

#### Initial Conditions

positions = m3t.get(pdb, coordinates=True)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)

#### Iterations Parameters

steps_simulation = 100000000 # 200 ns
steps_interval_saving = 2500 # 5 ps 
steps_interval_verbose = 500000 # 1 ns
steps_interval_checkpoint = 500000 # 1n

time_simulation = (steps_simulation*step_size).in_units_of(unit.picoseconds)
time_saving = (steps_interval_saving*step_size).in_units_of(unit.picoseconds)
time_verbose = (steps_interval_verbose*step_size).in_units_of(unit.picoseconds)
time_checkpoint = (steps_interval_checkpoint*step_size).in_units_of(unit.picoseconds)

log.write("\n")
log.write("Step size: {}\n".format(step_size))
log.write("Simulation time: {} ({} steps)\n".format(time_simulation , steps_simulation))
log.write("Saving time: {} ({} steps)\n".format(time_saving , steps_interval_saving))
log.write("Verbose time: {} ({} steps)\n".format(time_verbose , steps_interval_verbose))
log.write("Checkpoint time: {} ({} steps)\n".format(time_checkpoint , steps_interval_checkpoint))
log.write("\n")


#### Reporters

# Logfile

simulation.reporters.append(app.StateDataReporter(log, reportInterval=steps_interval_verbose,
                                                  progress=True, speed=True, step=True, time=True,
                                                  potentialEnergy=True, temperature=True,
                                                  volume=True, totalSteps=steps_simulation,
                                                  separator=", "))

# Observables

simulation.reporters.append(HDF5Reporter(trajectory_file, reportInterval=steps_interval_saving,
                                         coordinates=True, time=True, cell=True,
                                         potentialEnergy=True, kineticEnergy=True,
                                         temperature=True))

# Checkpoints

simulation.reporters.append(app.CheckpointReporter(checkpoint_file, steps_interval_checkpoint))

#### Running Simulation

start_simulation_realtime = realtime()

simulation.step(steps_simulation)

end_simulation_realtime = realtime()

#### Saving Finnal State

simulation.saveState(final_state_file)
simulation.saveCheckpoint(final_checkpoint_file)
m3t.convert(simulation,final_pdb_file)

#### Removing partial checkpoints
os.remove(checkpoint_file)

#### Closing all reporters but Log
simulation.reporters[1].close()

#### Moving files from temporal to current working dir:

for file_name in os.listdir(tmp_dir):
    file_tmp_path = os.path.join(tmp_dir, file_name)
    file_work_path = os.path.join(work_dir, file_name)
    shutil.move(file_tmp_path, file_work_path)

os.rmdir(tmp_dir)

#### Summary

end_realtime = realtime()
preparation_elapsed_realtime = (start_simulation_realtime - start_realtime)*unit.seconds
simulation_elapsed_realtime = (end_simulation_realtime - start_simulation_realtime)*unit.seconds
total_elapsed_realtime = (end_realtime - start_realtime)*unit.seconds

performance = 24 * (steps_simulation*step_size/unit.nanoseconds) / (simulation_elapsed_realtime/unit.hours)

log.write("\n")
log.write("End: "+formatted_local_time()+"\n")
log.write("\n")
log.write("****SUMMARY****\n")
log.write("\n")
log.write("Total time: "+formatted_elapsed_time(total_elapsed_realtime/unit.seconds)+"\n")
log.write("Preparation time: "+formatted_elapsed_time(preparation_elapsed_realtime/unit.seconds)+"\n")
log.write("Simulation time: "+formatted_elapsed_time(simulation_elapsed_realtime/unit.seconds)+"\n")
log.write("\n")
log.write("Simulation Performance: {:.3f} ns/day".format(performance)+"\n")
log.write("\n")

#### Closing Log
log.close()

