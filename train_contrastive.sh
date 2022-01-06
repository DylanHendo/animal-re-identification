#!/usr/bin/env bash

#PBS -N contrastive
#PBS -l gputype=M40
#PBS -l ngpus=1
#PBS -l ncpus=4
#PBS -l mem=18GB
#PBS -l walltime=01:30:00
#PBS -o /home/n10325701/output/stdout_contrastive.out
#PBS -e /home/n10325701/output/stderr_contrastive.out

###############################################
#
#
#  Display PBS info
#
#
###############################################
print_pbs_info(){
    echo ------------------------------------------------------
    echo -n 'Job is running on node '; cat $PBS_NODEFILE
    echo ------------------------------------------------------
    echo PBS: qsub is running on $PBS_O_HOST
    echo PBS: originating queue is $PBS_O_QUEUE
    echo PBS: executing queue is $PBS_QUEUE
    echo PBS: working directory is $PBS_O_WORKDIR
    echo PBS: execution mode is $PBS_ENVIRONMENT
    echo PBS: job identifier is $PBS_JOBID
    echo PBS: job name is $PBS_JOBNAME
    echo PBS: node file is $PBS_NODEFILE
    echo PBS: current home directory is $PBS_O_HOME
    echo PBS: PATH = $PBS_O_PATH
    echo ------------------------------------------------------

    # displaying some additional node info
    # is handy for debugging some things or know if you are
    # encountering any problematic nodes
    echo ""
    echo ------------------------------------------------------
    pbsnodeinfo | head -n 2
    pbsnodeinfo | grep $HOSTNAME
}



###############################################
#
#
#  Helper/Setup Functions
#
#
###############################################

load_modules(){
    #activate module environment
    #NOTE: a recent HPC update means that you shouldn't need
    #to do this anymore, but I have included as a sanity check
    source /etc/profile.d/modules.sh

    #load TF-python
    module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4

    # activate the virtual environment we need
    source ~/train/bin/activate
}

run_program(){
    python /home/n10325701/src/contrastive.py
}


###############################################
#
#
#  Running everything
#
#
###############################################

print_pbs_info
load_modules
run_program

