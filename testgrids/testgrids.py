###############################################################################
#                                                                             #
#                              *** testgrids ***                              #
#                                                                             #
# Description: Python 3 script for investigating the behaviour of CASTEP      #
#                  simulations with a varying FINE_GMAX parameter for a       #
#                  number of elements. The script takes care of generating    #
#                  test data, submitting it to SLURM, monitoring the queue,   #
#                  extracting and plotting the relevant data. To get help     #
#                  with usage of testgrids, consult the documentation or      #
#                  call testgrids with the --help flag.                       #
#                                                                             #
# Author: Alexander Liptak (Alexander.Liptak.2015@live.rhul.ac.uk)            #
# Supervisor: Prof Keith Refson (Keith.Refson@rhul.ac.uk)                     #
# Date: Summer 2019                                                           #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
#  Exit codes:                                                                #
#  ----------                                                                 #
#  00   : No errors encountered                                               #
#  ----------                                                                 #
#  01   : ImportError while importing a module                                #
#  02   : argument directory does not exist or is inaccessible                #
#  03   : argument file does not not exist or is inaccessible                 #
#  04   : TypeError when parsing argument                                     #
#  05   : argument out of bounds                                              #
#  06   : argument not one of possible options                                #
#  07   : CalledProcessError when loading CASTEP module                       #
#  ----------                                                                 #
#  10   : OSError while generating job data                                   #
#  11   : CalledProcessError while generating job data                        #
#  19   : uncaught exception while generating job data                        #
#  ----------                                                                 #
#  20   : CalledProcessError while running jobs                               #
#  29   : Uncaught exception while running jobs                               #
#  ----------                                                                 #
#  30   : OSError while reading job output data                               #
#  31   : PickleError when dumping array data to disk                         #
#  39   : Uncaught exception while reading job output data                    #
#  ----------                                                                 #
#  40   : OSError while cleaning up                                           #
#  49   : Uncaught exception while cleaning up                                #  
#  ----------                                                                 #
#  50   : OSError while opening file handle to pickle file                    #
#  51   : PickleError when loading pickle data into memory                    #
#  59   : Uncaught exception while loading arrays into memory                 #
#  ----------                                                                 #
#  69   : Uncaught exception while deleting unused semaphores                 #
#  ----------                                                                 #
#                                                                             #
###############################################################################

try:                    # import required modules
    from argparse import ArgumentParser, ArgumentTypeError, ArgumentError
    from os import mkdir, listdir
    from os.path import abspath, isdir, isfile
    from shutil import rmtree
    from subprocess import Popen, PIPE, CalledProcessError
    from sys import exit, modules
    from time import sleep
    from pickle import dump, load, PickleError
    import numpy as np
    
    from eosfit import BM
    from calcDelta import calcDelta

    import matplotlib   # import matplotlib first to select interactive backend
    if matplotlib.get_backend() in matplotlib.rcsetup.interactive_bk:
        interactive = True      # interactive backend already loaded
    else:               # attempt to load all supported inter. backends in order
        interactive = False     # will remain false if no inter. backends loaded
        for backend in matplotlib.rcsetup.interactive_bk:
            try:        # backend without required modules with raise ImportErr
                matplotlib.use(backend, warn=False, force=True)
                import matplotlib.pyplot as plt     # should fail here if unsup.
                interactive = True      # if here then inter. backend loaded ok
                print("[INFO] Switched to mpl backend {}".format(backend))
                break   # no need to carry on trying other backends
            except:     # all ImportErrors from pyplot will be caught here
                continue    # ignore them as we can try other backends
    
except ImportError as e:
    print("[ERR 01] A fatal ImportError has occured: {}".format(e))
    exit(1)
    
debug_mode = False          # if true, code will output additional info

###############################################################################
#            Contents of CASTEP cell and param files to be written            #
###############################################################################

param_file_data = '\n'.join(["task                : singlepoint",
                             "write_checkpoint    : none",
                             "write_cif_structure : false",
                             "write_bib           : false",
                             "opt_strategy        : speed",
                             "xc_functional       : {}",
                             "grid_scale          : {}",
                             "fine_grid_scale     : {}",
                             "num_proc_in_smp     : {}",
                             "calculate_stress    : {}",
                             "iprint              : {}",
                             "spin_polarized      : {}",
                             "spin                : {}",
                             "elec_energy_tol     : {} eV",
                             "basis_precision     : {}",
                             "perc_extra_bands    : {}",
                             "max_scf_cycles      : {}",
                             "mixing_scheme       : {}",
                             "mix_charge_amp      : {}",
                             "mix_spin_amp        : {}"])

###############################################################################
#         Functions for preparing, running and cleaning up after jobs         #
###############################################################################

def generate_job_data(elements, coarse_grid, fine_grid, job_name, cif_dir,
                      nodes, tasks, reserve, stress, ilevel, ps_pot, xc_func,
                      ee_tol, basis_prec, extra_bands, scf_cycles,
                      mixing_scheme, mix_charge_amp, mix_spin_amp,
                      kpoint_spacing, vol_scale):
    """Generates folders with .cell and .param files for all grid combinations
    
    Parameters:
    ----------
    elements       : List[str]
        List of .cif for every element to be simulated
    coarse_grid    : float
        Fixed size of coarse grid
    fine_grid      : List[float] or np.ndarray with dtype float
        List or array of all fine grid sizes to simulate
    job_name       : str
        The name of the job folder
    cif_dir        : str
        Absolute path to a folder contining .cif files
    nodes          : int
        Number of nodes that the job will use
    tasks          : int
        Number of tasks that one job will use
    reserve        : str
        Reservation string used when submitting jobs to SLURM
    stress         : bool
        Whether to perform stress calculations
    ilevel         : int 
        Verbosity level of the resulting output files
    ps_pot         : str
        Pseudopotential library to use for simulations
    xc_func        : str
        Exchange correlation functional to use for simulations
    ee_tol         : float
        Tolerance for acceptance of electronic minimisation energy
    basis_prec     : str
        The basis precision to use
    extra_bands    : float
        Percentage of extra bands to use in addition to the occupied ones
    scf_cycles     : int
        The maximum number of SCF cycles to perform
    mixing_scheme  : str
        Mixing scheme used in the density mixing procedure
    mix_charge_amp : float
        Mixing amplitude for the charge density in density mixing procedure
    mix_spin_amp   : float
        Mixing amplitude for the spin density in density mixing procedure
    kpoint_spacing : float
        Specifies the value of the kpoint_mp_spacing parameter
    vol_scale      : float
        The scale factor by which the cell volume should be adjusted
    
    Returns:
    ----------
    None"""

    try:
        print("[INFO] Generating job data ({} element(s), {} fine grid sizes, "
              "{} volumes)...".format(len(elements), len(fine_grid), 7))
        mkdir(job_name)  # create dir that will contain all data from this run
        if debug_mode: print("[DEBUG] created working dir {}".format(job_name))
        mkdir("{}/cell".format(job_name))   # create dir for master cell files
        for element in sorted(elements):    # each elem represents one job array
            symbol = element.split('.')[0]
            ferromagnetic = symbol in ["Co", "Ni", "Fe"]
            antiferromagnetic = symbol in ["Mn", "Cr", "O"]
            spin_polarized = ferromagnetic or antiferromagnetic
            if debug_mode:
                print("[DEBUG] symbol {}, spin polarized {}, ferromagnetic {}, "
                      "antiferromagnetic {}"
                      .format(symbol, spin_polarized, ferromagnetic,
                              antiferromagnetic))
            else:
                print("[INFO] Processing element {}".format(symbol), end='\r')
            
            # cif2cell will write a master .cell file for each element into the
            # job_name\cell that will be edited (volume scaling) and/or copied
            # into directories of .cell and .param files (individual jobs)
            if debug_mode:
                print("[DEBUG] calling cif2cell with cif file {}/{}"
                      .format(cif_dir, element))
            proc = Popen("{} cif2cell {}/{} -p castep"
                         .format(clean_env, cif_dir, element),
                        shell=True, cwd="{}/cell".format(job_name), stdout=PIPE)
            proc.wait()         # wait until process finishes, get return code
            if proc.returncode != 0:    # cif2cell call failed, cannot continue
                raise CalledProcessError(proc.returncode, proc.args)
            
            # now iterate through all fine_grid sizes and volume scaling factors
            # create directories in form of element_n with n (0 to fine_grid*7)
            # each containing volume-scaled .cell files and .param files
            for grid_idx, grid_val in enumerate(fine_grid):
                for vol_idx, vol_mult in enumerate(np.linspace(0.94, 1.06, 7)):
                    job_dir = "{}_{}".format(symbol, vol_idx+(grid_idx*7))
                    mkdir("{}/{}".format(job_name, job_dir))
                    if debug_mode:
                        print("[DEBUG] mkdir {}/{}".format(job_name, job_dir))
                      
                    # fill in .param file and save to job directory
                    with open("{0}/{1}/{1}.param".format(job_name, job_dir),
                              'w') as param_file:
                        param_file.write(param_file_data
                            .format(xc_func,        # exchange correlation func
                                    coarse_grid,    # GRID_SIZE
                                    grid_val,       # FINE_GRID_SIZE
                                    tasks,          # NUM_PROC_IN_SMP
                                    "true" if stress else "false",
                                    ilevel,         # ILEVEL
                                    "TRUE" if spin_polarized else "FALSE",
                                    "2" if ferromagnetic else "0",
                                    ee_tol,         # ELEC_ENERGY_TOL
                                    basis_prec,     # BASIS_PRECISION
                                    extra_bands,    # PERC_EXTRA_BANDS
                                    scf_cycles,     # MAX_SCF_CYCLES 
                                    mixing_scheme,  # MIXING_SCHEME
                                    mix_charge_amp, # MIX_CHARGE_AMP
                                    mix_spin_amp))  # MIX_SPIN_AMP
                    if debug_mode:
                        print("[DEBUG] wrote file {0}/{1}/{1}.param "
                              "(FINE_GRID_SCALE = {2})"
                              .format(job_name, job_dir, grid_val))
                    
                    # open output .cell file and input .cell master file
                    with open("{}/cell/{}.cell".format(job_name, symbol),
                              'r') as master_cell_file:
                        cell_data = master_cell_file.read()     # get all text
                        
                        # find indices of lines that begin and end LATTICE_CART
                        # extract BLOCK LATTICE_CART from cell data
                        # then replace all non-zero values with volume-scaled
                        # values formatted to 15 dp as in original cell data
                        cart_idx = [index for index, line
                                   in enumerate(cell_data.split('\n'))
                                   if "LATTICE_CART" in line]
                        old_cart_blk = '\n'.join(cell_data
                            .split('\n')[cart_idx[0]+2:cart_idx[1]])
                        cart_blk = old_cart_blk[:]
                        for val in cart_blk.split():
                            if float(val) != 0:
                                cart_blk = cart_blk.replace(
                                    val, "{0:.15f}".format(
                                        float(val)*vol_scale*np.cbrt(vol_mult)))
                        
                        # find indices of lines that begin and end POSITONS_FRAC
                        # extract BLOCK POSITIONS_FRAC from cell data
                        # if sample is antiferromagnetic, include SPIN=2 at the
                        # end of each line, where the sign alternates each line
                        frac_idx = [index for index, line
                                   in enumerate(cell_data.split('\n'))
                                   if "POSITIONS_FRAC" in line]
                        old_frac_blk = '\n'.join(cell_data
                            .split('\n')[frac_idx[0]+1:frac_idx[1]])
                        frac_blk = old_frac_blk[:]
                        if antiferromagnetic:
                            sign = 0  # controls +/-, %1 for Mn and Cr, %2 for O
                            for line in frac_blk.split('\n'):
                                frac_blk = frac_blk.replace(line,
                                                            line+" SPIN={}2"
                                    .format("-" if int(sign)%2 == 0 else "+"))
                                if symbol in ["Mn", "Cr"]:  # if elem Mn or Cr
                                    sign += 1           # alternate every point
                                else:                   # if elem is O
                                    sign += 0.5     # aternate every two points
                        
                        # find indices of lines that begin and end SPECIES_POT
                        # extract BLOCK SPECIES_POT from cell data
                        # find word that contains .usp
                        # remove comments and replace usp word with custom pot
                        pot_idx = [index for index, line
                                   in enumerate(cell_data.split('\n'))
                                   if "SPECIES_POT" in line]
                        old_pot_blk = '\n'.join(cell_data
                            .split('\n')[pot_idx[0]:pot_idx[1]+1])
                        usp_word = [word for word in old_pot_blk.split()
                                    if ".usp" in word][0]
                        pot_blk = (old_pot_blk
                                   .replace('#', '')
                                   .replace(usp_word, ps_pot))
                        
                        # find indices of lines that begin and end SYMMETRY_OPS
                        # extract BLOCK SYMMETRY_OPS from cell data
                        # whole block will be replaced with symmetry_generate
                        symm_idx = [index for index, line
                                   in enumerate(cell_data.split('\n'))
                                   if "SYMMETRY_OPS" in line]
                        old_symm_blk = '\n'.join(cell_data
                            .split('\n')[symm_idx[0]:symm_idx[1]+1])
                        symm_blk = "symmetry_generate"
                        
                        # replace old BLOCK LATTICE_CART with updated one
                        # replace old BLOCK POSITIONS_FRAC with updated one
                        # replace old BLOCK SPECIES_POT with updated one
                        # replace old BLOCK SYMMETRY_OPS with updated one
                        # write returned cell data to new .cell file
                        with open("{0}/{1}/{1}.cell".format(job_name, job_dir),
                              'w') as cell_file:
                            cell_file.write(cell_data
                                            .replace(old_cart_blk, cart_blk)
                                            .replace(old_frac_blk, frac_blk)
                                            .replace(old_pot_blk, pot_blk)
                                            .replace(old_symm_blk, symm_blk)
                                            +"kpoint_mp_spacing={}\n"
                                            .format(kpoint_spacing))
                        if debug_mode:
                            print("[DEBUG] wrote file {0}/{1}/{1}.cell "
                                  "(vol = {2} V0)"
                                  .format(job_name, job_dir, vol_mult))
            
            # create array job file (one per element) from job template
            # append the absolute path of job file to list (will be returned)
            with open("template.job", 'r') as job_template, open("{}/{}.job"
                .format(job_name, symbol), 'w') as job_file:
                job_file.write(job_template.read()
                    .format(symbol, nodes, tasks, len(fine_grid)*7, reserve))
            if debug_mode:
                print("[DEBUG] wrote file {}/{}.job".format(job_name, symbol))    

        print("[INFO] Prepared {} job array(s) ({} jobs in total)"
              .format(len(elements), len(elements)*len(fine_grid)*7))

    except OSError as e:
        print("[ERR 10] OSError: Cannot write to directory {}/{}\n (Verifiy "
              "(that the directory has enough free space and adequate write "
              "permissions)".format(job_name, job_dir))
        print("Exception data: {}".format(e))
        exit(10)
    except CalledProcessError as e:
        print("[ERR 11] CalledProcessError: Call to cif2cell failed.")
        print("Exception data: {}".format(e))
        exit(11)
    except Exception as e:
        print("[ERR 19] Uncaught exception: {}".format(e))
        exit(19)

###############################################################################

def run_jobs(job_no, queue_lim, job_name, poll_wait):
    """Submits all jobs to SLURM and waits for completion
    
    Parameters:
    ----------
    job_no    : int
        Number of jobs per job array
    queue_lim : int
        Maximum number of job arrays to submit to SLURM at a time
    job_name  : str
        The name of the job folder
    poll_wait : int
        Number of seconds to wait between squeue polls

    Returns:
    ----------
    None"""
    
    # compile a list of all .job files in job folder
    jobs = sorted([abspath("{}/{}".format(job_name, job))  # abs path to .job
                  for job in listdir(job_name) if job.endswith(".job")])
    pids = []                   # list of PIDs of job arrays submitted to SLURM
    total_jobs = len(jobs)*job_no           # total number of individual jobs
    if debug_mode:
        print("[DEBUG] using job files: {}".format(jobs))
    
    try:
        print("[INFO] Submitting jobs to SLURM ...")
        while len(jobs) != 0:   # if there are any jobs waiting to be submitted
            
            # select as many remaining jobs as fit into queue limit
            # submit jobs to SLURM using sbatch in a clean evironment
            # record PID of job array
            for job in jobs[:queue_lim if queue_lim < len(jobs) else len(jobs)]:
                proc = Popen("{} sbatch {}".format(clean_env, job),
                             shell=True, cwd=job_name, stdout=PIPE)
                proc_output = proc.communicate()[0].decode('ascii')
                if proc.returncode != 0:
                    raise CalledProcessError(proc.returncode, proc.args)
                pids.append(proc_output.rstrip().split()[-1])
            
            # remove submitted job arrays from job list
            jobs = jobs[queue_lim if queue_lim < len(jobs) else len(jobs):]
            
            # monitor SLURM queue and wait until all submitted jobs are finished
            while len(pids) != 0:
                # get a processes that are either running or in SLURM queue
                squeue = Popen("squeue", shell=True, stdout=PIPE)
                running_procs = squeue.communicate()[0].decode('ascii')
                if squeue.returncode != 0:
                    raise CalledProcessError(proc.returncode, proc.args)
                
                # remove job array PIDs that are no longer in SLURM queue
                new_pids = []
                for pid in pids:
                    if pid in running_procs:
                        new_pids.append(pid)
                if len(new_pids) == 0: break
                pids = new_pids
                
                # count jobs in job arrays in SLURM queue
                # if job ID has brackets, count number of jobs it encompasses
                # otherwise just increment number of individual jobs
                in_slurm = 0
                jobids = [jobid.split('_')[1] for jobid in running_procs.split()
                          if any([pid in jobid for pid in pids])]
                for jobid in jobids:
                    if '-' in jobid:
                        start, end = jobid[1:-1].split('-')
                        in_slurm += int(end) - int(start)
                    else:
                        in_slurm += 1
                
                print("[INFO] {} in queue, {} in SLURM, {} completed [{:.1f}%] "
                    .format(len(jobs) * job_no, in_slurm,
                            total_jobs - (len(jobs) * job_no) - in_slurm,
                            (total_jobs - (len(jobs) * job_no) - in_slurm) * 100
                            /total_jobs),
                      end='\r')
                sleep(poll_wait)    # wait poll_wait seconds before rechecking

    except CalledProcessError as e:
        print("[ERR 20] CalledProcessError: Returned with non-zero exit code.")
        print("Exception data: {}".format(e))
        exit(20)    
    except Exception as e:
        print("[ERR 29] Uncaught exception: {}".format(e))
        exit(29)
            
###############################################################################

def analyse_data(job_name, fine_grid, stress):
    """Reads, extracts and analyses data from .castep files
    
    Parameters:
    ----------
    job_name  : str
        The name of the job folder
    fine_grid : np.ndarray(dtype=float)
        Array of all fine grid sizes to simulate
    stress    : bool
        Whether stress calculations have been carried out
        
    Returns:
    ----------
    arrays    : dict{'element': dict {'array_name': np.ndarray}}
        Dictionary of elements as keys and dictionaries of arrays as values with
        array names as keys"""
    
    arrays = {}

    try:
        print("[INFO] Extracting data from .castep files ...                  ")
        # get a list of uniqe elements in job directory
        elements = set([folder.split('_')[0] for folder in listdir(job_name)
                        if '.' not in folder and folder != "cell"])
        if debug_mode:
            print("[DEBUG] found folders for elements: {}".format(elements))
        
        # go through all folders in job directory, and for each element ...
        for elem in elements:   # find all folders corresponding to that element

            cell_vols = []      # Cell volumes
            scf_iters = []      # Number of SCF iterations to converge energy
            fin_enrgs = []      # Final converged energy
            pressures = []      # Pressure
            tot_times = []      # Total simulation time
            
            f_grid_dm = []      # Fine grid dimensions
            real_latt = []      # Real lattice vectors
            
            for job_dir in sorted([folder for folder in listdir(job_name)
                                   if '.' not in folder and elem+"_" in folder],
                                  key=lambda elem: int(elem.split('_')[-1])):
                # if there are any error files, simulation likely failed
                if any([".err" in file for file in 
                        listdir("{}/{}".format(job_name, job_dir))]):
                    print("[WARNING] found .err file in dir: {}/{}"
                              .format(job_name, job_dir))
                    # in that case, just set the result as None and continue
                    for data in [cell_vols, scf_iters, fin_enrgs, pressures,
                                 tot_times, f_grid_dm, real_latt]:
                        data.append(np.nan)
                    continue
                
                with open("{0}/{1}/{1}.out".format(job_name, job_dir), 
                          'r') as out_file:
                    # if there is any error in the .out file
                    out_data = out_file.read()
                    if "Error" in out_data or "Killed" in out_data:
                        print("[WARNING] job {0}/{1}/{1} has encountered an "
                              "error or been killed".format(job_name, job_dir))
                        # in that case, just set the result as None and continue
                        for data in [cell_vols, scf_iters, fin_enrgs, pressures,
                                    tot_times, f_grid_dm, real_latt]:
                            data.append(np.nan)
                        continue
                                    
                # if the simulation did not fail, open the .castep file
                with open("{0}/{1}/{1}.castep".format(job_name, job_dir),
                          'r') as castep_file:
                    # variable "looking_for" signals what value is the next
                    # required one, this is done in order of appearance in the
                    # file, so that it doesn not need to be read more than once
                    looking_for = "fine_grid_dims"
                    scf_iter = -6       # offset, as there are 6 nan scf lines
                    found_rl_lines = -1     # number real lattice lines found
                    stress_tensor_found = False     # only look for pressure ...
                    warnings = 0        # print warning if this is bigger than 5
                    # values after the symmetrised stress tensor has been found
                    
                    for line in castep_file:    # go through file line by line
                        if looking_for == "nothing": break  # exit if all found

                        # if a warning has been found on a line, report it and
                        # go to the next line
                        if "Warning" in line:
                            warnings += 1
                            if warnings > 5:
                                print("[WARNING] over 5 warnings detected in "
                                      "file {0}/{1}/{1}.castep"
                                      .format(job_name, job_dir))
                            continue
                    
                        # if looking for fine grid dimensions in line, grab the 
                        # 3 values corresponding to the dimensions in each of
                        # the 3 lattice vector directions
                        if looking_for == "fine_grid_dims":
                            if "Fine grid dimensions" in line:
                                f_grid_dm.append([int(d) for d in 
                                                  line.split()[3:6]])
                                looking_for = "real_lattice"
                            continue
                        
                        # if looking for real lattice vectors, wait until
                        # real_lattice appears in text, then append a new empty
                        # array to the list of real lattice vectors, which will
                        # be filled in the next few lines
                        if looking_for == "real_lattice":
                            if "Real Lattice" in line:
                                found_rl_lines = 0
                                real_latt.append([])
                                continue
                            if -1 < found_rl_lines < 3:
                                real_latt[-1].append((np.sqrt(np.sum(np.square(
                                    [float(v) for v in line.split()[:3]])))))
                                found_rl_lines += 1
                            if found_rl_lines == 3:
                                looking_for = "cell_volume"
                            continue
                    
                        # if looking for cell volume, find line and append float
                        if looking_for == "cell_volume":
                            if "Current cell volume" in line:
                                cell_vols.append(float(line.split()[4]))
                                looking_for = "total_energy"
                            continue
                    
                        # if looking for total energy, count all occurences of
                        # SCF corresponding to the number of SCF iterations
                        # stop when final energy is found, look for pressure
                        if looking_for == "total_energy":
                            if "<-- SCF" in line: scf_iter += 1
                            if "Final energy" in line:
                                scf_iters.append(scf_iter)
                                fin_enrgs.append(float(line.split()[4]))
                                looking_for = "pressure"
                            continue
                        
                        # continue going through file line by line, looking for
                        # pressure text, record it when found, look for time
                        if looking_for == "pressure":
                            if stress:
                                if "Symmetrised Stress Tensor" in line:
                                    stress_tensor_found = True
                                if "Pressure:" in line and stress_tensor_found:
                                    pressures.append(float(line.split()[2]))
                                    looking_for = "total_time"
                            else:
                                pressures.append(np.nan)
                                looking_for = "total_time"
                            continue
                        
                        # same for total time, but if found, stop looking
                        if looking_for == "total_time":
                            if "Total time" in line:
                                tot_times.append(float(line.split()[3]))
                                looking_for = "nothing"
                    
                    # if not all items were found, add np.nan for correct length
                    if looking_for != "nothing":
                        objs = [f_grid_dm, real_latt, cell_vols, scf_iters,
                                fin_enrgs, pressures, tot_times]
                        if looking_for == "fine_grid_dims": idx = 0
                        if looking_for == "real_lattice":   idx = 1
                        if looking_for == "cell_volume":    idx = 2
                        if looking_for == "total_energy":   idx = 3
                        if looking_for == "pressure":       idx = 5
                        if looking_for == "total_time":     idx = 6
                        for data in objs[idx:]: data.append(np.nan)
            
            if debug_mode:
                print("[DEBUG] extracted data for {}, tabulating:".format(elem))
                print("*"*100+"\n")
                for data_array, data_label in zip(
                    [scf_iters, fin_enrgs, pressures, tot_times],
                    ["SCF ITERS", "FINAL ENRGS", "PRESSURES", "TOTAL TIMES"]):
                    fmt = max([len("{:.5f}".format(i)) for i in data_array])
                    if fmt < 4: fmt = 4     # if below minimum, set to minimum
                    print(("{:16} "+"| {:^{width}.2f} "*7)
                          .format(data_label, *np.linspace(0.94, 1.06, 7),
                                  width=fmt))
                    print("-"*(37+7*fmt))
                    for f in range(len(fine_grid)):
                        print(("FINE_GRID = {:.2f} "+"| {:{width}.5f} "*7)
                              .format(fine_grid[f], *data_array[f*7:(f+1)*7],
                                      width=fmt))
                    print("\n"+"*"*100+"\n")

            eos_results = []            # results from eosfit function
            residuals   = []            # resiudals from eosfit function
            deltas  = []                # results from calcDelta function
            
            # group all volumes and energies into 7s, iterate over them
            # calculating the volume, bulk modulus and bulk derivative from the
            if debug_mode:
                print("[DEBUG] fitting EOS for {} ".format(elem))
                print("*"*100+"\n")
            for volumes, energies in zip(
                    np.array(cell_vols).reshape(len(fine_grid), 7),
                    np.array(fin_enrgs).reshape(len(fine_grid), 7)):
                fmt = max([len("{:.5f}".format(i)) for i in energies])
                if debug_mode:
                    for label, values in zip(["volumes:", "energies:"],
                                             [volumes, energies]):
                        print(" | ".join(["{:15}", *["{:{width}.5f}"]*7])
                              .format(label, *values, width=fmt))
                    print("-"*(37+7*fmt))
                try:
                    vol, bulk_mod, bulk_der, resid = BM(volumes, energies)
                    if not isinstance(vol, (np.float64, np.float32)):
                        if debug_mode:
                            print("[DEBUG] discarded as EOS fitting function "
                                  "returned a complex or None type (not float)")
                        raise Exception
                    if bulk_der < 0:
                        if debug_mode:
                            print("[DEBUG] discarded as EOS fitting function "
                                  "returned a negative bulk modulus derivative")
                        raise Exception
                except:
                    vol, bulk_mod, bulk_der = [np.nan]*3
                    resid = [np.nan]    # to be consistent with BM formatting
                if debug_mode:
                    for label, value in zip(["V0:", "bulk_modulus:",
                                             "bulk mod deriv:", "B0:",
                                             "residual:"],
                                             [vol, bulk_mod, bulk_der,
                                              bulk_mod*1.60217733e-19*1e21,
                                              resid[0]]):
                        print("{:15} | {}".format(label, value))
                    print("\n"+"*"*100+"\n")
                eos_results.append(np.array(
                    [(elem, vol, bulk_mod*1.60217733e-19*1e21, bulk_der)], 
                    dtype={'names': ('element', 'V0', 'B0', 'BP'),
                           'formats': ('S2', np.float, np.float, np.float)}))
                residuals.append(resid)
            
            # for all values in eos_results, calculate delta values
            if debug_mode:
                print("[DEBUG] tabulating deltas for {}:".format(elem))
                print("*"*50+"\n")
                print("  Delta   | relDelta  |  Delta1")
                print("-"*40)
            for eos in eos_results:
                if not any([np.isnan(i) for i in eos[0]
                        if not isinstance(i, np.bytes_)]):
                    delta = calcDelta(eos, eos_results[-1], [elem.encode()],
                                      False)
                else:
                    delta = [[np.nan]]*3
                deltas.append(delta)
                if debug_mode:
                    print((" | ".join(["{:^9.5f}"]*3))
                          .format(*[d[0] for d in deltas[-1]]))
            if debug_mode: print("\n"+"*"*50+"\n")
            
            # construct dictionary of all arrays per element
            elem_arrays = {'fine_grid': fine_grid,
                           'deltas'   : np.array(deltas,    dtype=np.float32),
                           'cell_vols': np.array(cell_vols, dtype=np.float32),
                           'scf_iters': np.array(scf_iters, dtype=np.float32),
                           'fin_enrgs': np.array(fin_enrgs, dtype=np.float32),
                           'pressures': np.array(pressures, dtype=np.float32),
                           'tot_times': np.array(tot_times, dtype=np.float32),
                           'fine_gmax': np.array(
                               [np.min(np.divide(np.multiply(np.pi, dim), cell))
                                for dim, cell in zip(f_grid_dm, real_latt)],
                               dtype=np.float32)}
            
            arrays[elem] = elem_arrays  # add per-element dict to job dict

            # draw table of information from arrays and save to text file
            with open("{}.table.txt".format(job_name), 'a') as file:
                file.write("*"*131+"\n"
                           +"*{:^129}*\n".format("== {} ==".format(elem))
                           +"*"*131+"\n"
                           +("* {:^10} * {:^88} "+"* {:^10} "*2+"*\n")
                            .format("FINE_GMAX", "Cell volume as percentage of "
                                    "V0 (Final energy [eV], Pressure [GPa], "
                                    "Simulation time [s])", "Delta", "B0")
                           +("* {:^10} "+"*"*91+"* {:^10} "*2+"*\n")
                            .format("FINE_SCALE", "relDelta", "B\'")
                           +("* {:^10} "*10+"*\n")
                            .format("V0", "94%", "96%", "98%", "100%", "102%",
                                    "104%", "106%", "Delta1", "residual")
                           +"*"*131+"\n")
                for row_index in range(len(elem_arrays['fine_grid'])):
                    file.write(("* {:^10.5f} "+"* {:^10.4f} "*7
                                +"* {:^10.5f} "*2+"*\n")
                        .format(elem_arrays['fine_gmax'][row_index*7+3],
                                *elem_arrays['fin_enrgs']
                                            [row_index*7:(row_index+1)*7],
                                *elem_arrays['deltas'][row_index][0],
                                *eos_results[row_index]['B0']))
                    file.write(("* {:^10.5f} "*10+"*\n")
                        .format(elem_arrays['fine_grid'][row_index],
                                *elem_arrays['pressures']
                                            [row_index*7:(row_index+1)*7],
                                *elem_arrays['deltas'][row_index][1],
                                *eos_results[row_index]['BP']))
                    file.write(("* {:^10.5f} "*10+"*\n")
                        .format(*eos_results[row_index]['V0'],
                                *elem_arrays['tot_times']
                                            [row_index*7:(row_index+1)*7],
                                *elem_arrays['deltas'][row_index][2],
                                *residuals[row_index]))
                    file.write("*"*131+"\n")
                file.write("\n")
            if debug_mode: print("[DEBUG] appended to file {}.table.txt"
                                 .format(job_name))

        # dump dictionary of all elements and arrays into pickle file
        print("[INFO] Pickling data for later use ...                         ")
        with open("{}.pickle".format(job_name), 'wb') as file:
            dump(arrays, file)
        if debug_mode: print("[DEBUG] wrote file {}.pickle".format(job_name))

        # machine readable file of delta values
        with open("{}.deltas.txt".format(job_name), 'w') as file:
            file.write("#"+("\t".join(["Elem", "FINE_GRID_SCALE", "FINE_GMAX",
                                       "Delta", "relDelta", "Delta1"]))+"\n")
            for elem in arrays:
                for i in range(len(arrays[elem]["fine_grid"])):
                    file.write("\t".join([elem,
                                         str(arrays[elem]["fine_grid"][i]),
                                         str(arrays[elem]["fine_gmax"][i]),
                                         *[str(d[0]) for d in 
                                           arrays[elem]['deltas'][i]],
                                         "\n"]))
        if debug_mode: print("[DEBUG] wrote file {}.deltas.txt"
                             .format(job_name))
                                        
    except OSError as e:
        print("[ERR 30] OSError: unable to read file {}.out\n This most likely "
              "means that the CASTEP job was terminated prematurely, or you "
              "do not have the permissions to read/write to taht file.\n"
              "Check the .out and .err files for more info.".format(job_name))
        print("Exception data: {}".format(e))
        exit(30)
    except PickleError as e:
        print("[ERR 31] PickleError: unable to dump data to {}.pickle"
              .format(job_name))
        print("Exception data: {}".format(e))
        exit(31)
    except Exception as e:
        print("[ERR 39] Uncaught exception: {}".format(e))
        exit(39)
    
    return arrays
        
###############################################################################

def plot_data(arrays, job_name, save_fig):
    """Plots data from arrays, per element
    
    Parameters:
    ----------
    arrays    : dict{'element': dict {'array_name': np.ndarray}}
        Dictionary of elements as keys and dictionaries of arrays as values with
        array names as keys
    job_name  : str
        The name of the job folder
    save_fig  : bool
        Save figure instad of displaying it on screen
    
    Returns:
    ----------
    None"""
    
    # if plotting figures, print information
    # if save_fig is used or no interactive backend is loaded, figs will save
    # check if figs folder exists, if it doesn't then create it
    # check if folder for current job name exits in figs, if it doesn't then ...
    # ... create it, if it does then skip saving figures because they exist
    if interactive and not save_fig:
        print("[INFO] Plotting data ...                                       ")
    else:
        print("[INFO] Saving plots ...                                        ")
        if not isdir("figs"):
            if debug_mode:
                print("[DEBUG] mkdir figs")
            mkdir("figs")
        if isdir("figs/{}".format(job_name)):
            print("[INFO] Figures already exist, skipping ...")
            return
        else:
            if debug_mode:
                print("[DEBUG] mkdir figs/{}".format(job_name))
            mkdir("figs/{}".format(job_name))
        
    
    # if job_name was a path to pickle, extract filename from path
    if any([string in job_name for string in ['/', '.']]):
        job_name = job_name.split('/')[-1].split('.')[0]
        if debug_mode: print("[DEBUG] set job name to {}".format(job_name))
    
    for elem in sorted(arrays):
        # if plotting module is loaded, plots can be generated
        if "matplotlib.pyplot" in modules:
            fig = plt.figure(figsize=[8, 6])     # figure for delta values
            fig.suptitle(elem)
            axes = fig.subplots(2, 2, sharex='col', sharey='row') 
            
            # set axes labels
            axes[0][0].set_ylabel("Delta value (coarse)")
            axes[1][0].set_ylabel("Delta value (fine)")
            axes[1][0].set_xlabel("FINE_GRID_SIZE")
            axes[1][1].set_xlabel("FINE_GMAX")
            
            # set axes limits
            axes[0][0].set_ylim(0, 10)
            axes[1][0].set_ylim(0, 1)
            
            # plot axes data
            for cl in axes:
                for idx, lbl in enumerate(['Delta', 'relDelta', 'Delta1']):
                    cl[0].plot(arrays[elem]['fine_grid'],
                               arrays[elem]['deltas'][:, idx],
                               label=lbl, ls='-', lw=1, marker='.', ms=5)
                    cl[1].plot(*zip(*sorted(zip(arrays[elem]['fine_gmax'][3::7],
                               arrays[elem]['deltas'][:, idx]))),
                               label=lbl, ls='-', lw=1, marker='.', ms=5)
            
            axes[0][1].legend()  # legend in top right corner of top right axis
            plt.tight_layout()   # compact plot elements using tight layout
            
            # if an interactive backend is loaded and we are not only saving
            if interactive and not save_fig:
                plt.show()       # show plot on screen
            else:                # otherwise save the picture figure folder
                if debug_mode:
                    print("[DEBUG] saving figure to figs/{}/{}"
                          .format(job_name, elem))
                plt.savefig("figs/{}/{}".format(job_name, elem))
                plt.close('all')

###############################################################################

def load_data(pickle_file):
    """Loads arrays from pickle file
    
    Parameters:
    ----------
    pickle_file : str
        Path or name of the file to unpickle
    
    Returns:
    ----------
    arrays      : dict{'element': dict {'array_name': np.ndarray}}
        Dictionary of elements as keys and dictionaries of arrays as values with
        array names as keys"""
    
    try:
        print("[INFO] Loading pickled data ...                                ")
        if debug_mode: print("[DEBUG] reading {}.pickle".format(pickle_file))
        with open(pickle_file, 'rb') as file:
            return load(file)
    except OSError as e:
        print("[ERR 50] OSError: unable to read file {}".format(pickle_file))
        print("Exception data: {}".format(e))
        exit(50)
    except PickleError as e:
        print("[ERR 51] PickleError: unable to load data from {}"
              .format(pickle_file))
        print("Exception data: {}".format(e))
        exit(51)
    except Exception as e:
        print("[ERR 59] Uncaught exception: {}".format(e))
        exit(59)

###############################################################################

def cleanup_wkdir(job_name):
    """Cleans up all temp files from previous jobs
    
    Parameters:
    ----------
    job_name : str
        The name of the job folder to clean up

    Returns:
    ----------
    None"""

    try:
        print("[INFO] Performing initial cleanup ...")
        for entry in listdir('./'):     # go through all files in CWD
            if job_name in entry and isdir(entry):  # if folder matches job name
                rmtree(entry)           # recursively delete folder
                if debug_mode:
                    print("[DEBUG] cleared working directory: {}".format(entry))
    except OSError as e:
        print("[ERR 40] OSError: cannot delete job folder. Check permissions.")
        print("Exception data: {}".format(e))
        exit(40)
    except Exception as e:
        print("[ERR 49] Uncaught exception: {}".format(e))
        exit(49)
        
###############################################################################

def cleanup_semaphores():
    """Sends cleanup script as a job to idle nodes to clear unused semaphores
    
    Parameters:
    ----------
    None

    Returns:
    ----------
    None"""
    
    print("[INFO] Deleting unused semaphores ...")
    try:
        # run sinfo to get list of nodes, collect output as a table of nodes
        # with different statuses, and select nodes with idle status
        sinfo = Popen("sinfo", shell=True, stdout=PIPE)
        node_info = sinfo.communicate()[0].decode('ascii').split("\n")[1:-1]
        idle_nodes = [line.split()[5] for line in node_info
                      if line.split()[4] == "idle"]
        # for all idle nodes, execute ipcclean script on node
        for node in idle_nodes:
            proc = Popen("srun --nodelist {} --ntasks-per-node 1 ipcclean"
                         .format(node), shell=True, stdout=PIPE, stderr=PIPE)
            if debug_mode:      # record number of delted semaphores
                print("[DEBUG] deleted {} semaphores from idle nodes"
                        .format(proc.communicate()[0].decode('ascii')
                                .count("resource(s) deleted")))
    except Exception as e:
        print("[ERR 69] Uncaught exception: {}".format(e))
        exit(69)

###############################################################################
#      Functions for setting up argument parser and verifying arguments       #
###############################################################################

def setup_argparse():
    """Sets up argument parser

    Parameters:
    ----------
    None

    Returns:
    ----------
    parser : argparse.ArgumentParser
        Argument parser"""

    def _str2bool(arg):
        """Internal function for parsing booleans"""
        if isinstance(arg, bool):   # if arg is already a bool
            return arg              # just return the arg
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True             # if it is True-like, return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False            # if it is False-like, return False
        else:                       # otherwise throw error
            print("[Err 04] TypeError: {} cannot be interpreted as a boolean"
                  .format(arg))
            exit(4)

    parser = ArgumentParser(description="testgrids: script for automating the "
                            "calculation of delta values for CASTEP runs with "
                            "a varying FINE_GMAX parameter")
    parser.add_argument('-ba', '--extra_bands',    type=float,     default=100.0,   help="Percentage of extra bands to use in addition to occupied bands")
    parser.add_argument('-bp', '--basis_prec',     type=str,       default='FINE',  help="Basis precision to use")
    parser.add_argument('-ca', '--castep',         type=str,       default='',      help="CASTEP module to use")
    parser.add_argument('-cl', '--clean',          type=_str2bool, default=False,   help="Clean up after jobs")
    parser.add_argument('-cf', '--cif_dir',        type=str,       default='',      help="Directory of CIF files to use")
    parser.add_argument('-d',  '--debug',          type=int,       default=0,       help="Jump to code stage <d> (any nonzero will print debug output)")
    parser.add_argument('-ee', '--elec_enrg_tol',  type=float,     default=1e-5,    help="Tolerance for acceptance of electronic minimisation energy")
    parser.add_argument('-el', '--element',        type=str,       default='',      help="Simulate only the specified element")
    parser.add_argument('-f0', '--fine_grid_min',  type=float,     default=2.0,     help="Fine grid minimum size")
    parser.add_argument('-f1', '--fine_grid_max',  type=float,     default=6.0,     help="Fine grid maximum size")
    parser.add_argument('-fs', '--fine_grid_step', type=int,       default=9,       help="Number of fine grid sizes to sample")
    parser.add_argument('-g',  '--grid',           type=float,     default=1.75,    help="Standard grid fixed size")
    parser.add_argument('-i',  '--ilevel',         type=int,       default=2,       help="Verbosity level of .castep files")
    parser.add_argument('-k',  '--kpoint_spacing', type=float,     default=0.2,     help="Specifes the k-point spacing to use with the .cell file")
    parser.add_argument('-l',  '--load',           type=str,       default='',      help="Path to the pickle file to load")
    parser.add_argument('-mc', '--mix_charge_amp', type=float,     default=0.5,     help="Mixing amplitude for the charge density in density mixing procedure")
    parser.add_argument('-mi', '--mixing_scheme',  type=str,       default='PULAY', help="Mixing scheme used in the density mixing procedure")
    parser.add_argument('-ms', '--mix_spin_amp',   type=float,     default=0.5,     help="Mixing amplitude for the spin density in density mixing procedure")
    parser.add_argument('-na', '--name',           type=str,       default='tmp',   help="Job name")
    parser.add_argument('-no', '--nodes',          type=int,       default=1,       help="Number of nodes to use")
    parser.add_argument('-po', '--poll_wait',      type=int,       default=1,       help="Number of seconds to wait between squeue checks")
    parser.add_argument('-pr', '--primitive',      type=_str2bool, default=False,   help="Use primitive unit cells instead of the ground state structure")
    parser.add_argument('-ps', '--pseudo_pot',     type=str,       default='C19',   help="Pseudopotental library to use")
    parser.add_argument('-q',  '--queue',          type=int,       default=10,      help="Maximum numbers of jobs to submit to SLURM queue")
    parser.add_argument('-r',  '--reserve',        type=str,       default='',      help="SLURM reservation string")
    parser.add_argument('-sa', '--save',           type=_str2bool, default=True,    help="Save images instead of displaying them on screen")
    parser.add_argument('-sc', '--max_scf_cycles', type=int,       default=100,     help="Maximum number of SCF cycles to perform")
    parser.add_argument('-st', '--stress',         type=_str2bool, default=False,   help="Perform stress calculations")
    parser.add_argument('-t',  '--tasks',          type=int,       default=20,      help="Number of tasks to use")
    parser.add_argument('-v',  '--volume_scale',   type=float,     default=1.0,     help="Factor by which to scale cell volume")
    parser.add_argument('-x',  '--xc_functional',  type=str,       default="PBE",   help="Functional used to calculate the exchange-correlation potential")

    return parser

###############################################################################

def parse_args(parser):
    """Parses arguments and verifies them for validity

    Parameters:
    ----------
    parser : argparse.ArgumentParser
         Argument parser
    
    Returns:
    ----------
    args   : argparse.Namespace
         Namespace object containing parsed arguments"""

    args = parser.parse_args()      # collect all args from arg parser
    
    if args.load != "":             # if load has been selected
        if not isfile(args.load):   # check if it corresponds to a valid file
            print("[Err 03] Argument does not correspond to a valid or "
              "accessible file: {}".format(args.load))
            exit(3)
        else:                       # if it does, ignore the rest of the args
            return args

    try:
        # try if it is possible to load castep module
        if args.castep == '': args.castep = "CASTEP/19.1.1-foss-2018a"
        proc = Popen("module purge && module load {}".format(args.castep),
                     shell=True, stdout=PIPE)
        proc.wait()         # wait until process finishes, get return code
        if proc.returncode != 0:    # module load failed, cannot continue
            raise CalledProcessError(proc.returncode, proc.args)
    except CalledProcessError as e:
        print("[ERR 20] CalledProcessError: Unable to load CASTEP module.")
        print("Exception data: {}".format(e))
        exit(7)
    
    # if cif_dir has been selected, check that it is valid, otherwise throw err
    if args.cif_dir != '' and not isdir(args.cif_dir):
        print("[Err 02] Argument does not correspond to a valid or accessible "
              "path: {}".format(args.cif_dir))
        exit(2)
    
    # if cif_dir has not been selected, set it to one of the two cif paths
    if args.cif_dir == '':
        if args.primitive:
            args.cif_dir = abspath('./primCIFs')  # if using primitive unit cell
        else:
            args.cif_dir = abspath('./CIFs')      # otherwise (ground state str)
    else:   # if cif_dir has been selected and valid, set it to absolute path
        args.cif_dir = abspath(args_cif_dir)
        
    if args.debug != 0:
        global debug_mode
        debug_mode = True  # if nonzero debug, enable debug mode
    
    # if element is selected and does not correspond to a valid element from
    # the now-selected cif_dir, throw error
    if args.element != '' and not any([args.element in f for f 
                                       in listdir(args.cif_dir)]):
        print("[Err 03] Argument does not correspond to a valid or accessible "
              "file: {}/{}.cif".format(args.cif_dir, args.element))
        exit(3)
    
    # check that basis precision is one of the allowed options
    if args.basis_prec.upper() not in ["COARSE", "MEDIUM", "FINE", "PRECISE", 
                                       "EXTREME"]:
        print("[Err 06] Argument basis_prec is not one of possible options: "
              "[COARSE, MEDIUM, FINE, PRECISE, EXTREME]")
        exit(6)
    
    # check that the mixing scheme is one of the four supported ones
    if args.mixing_scheme.upper() not in ["BROYDEN", "KERKER", "LINEAR",
                                          "PULAY"]:
        print("[Err 06] Argument mixing_scheme is not one of possible options: "
              "[BROYDEN, KERKER, LINEAR, PULAY]")
        exit(6)
    
    # check that the pseudopotential function is one of three supported ones
    if args.pseudo_pot.upper() not in ["C19", "QC5", "NCP19"]:
        print("[Err 06] Argument pseudo_pot is not one of possible options: "
              "[C19, QC5, NCP19]")
        exit(6)
    
    # check that exchange correlation functional is one of the supported ones
    if args.xc_functional.upper() not in ["LDA", "PW91", "PBE", "PBESOL",
                                          "RPBE", "WC", "BLYP", "LDA-C2",
                                          "LDA-X", "ZERO", "HF", "PBE0",
                                          "B3LYP", "HSE03", "HSE06", "EXX-X",
                                          "HF-LDA", "EXX", "EXX-LDA", "SHF",
                                          "SX", "SHF-LDA", "SX-LDA", "WDA",
                                          "SEX", "SEX-LDA"]:
        print("[Err 06] Argument xc_functional is not one of possible options: "
              "[LDA, PW91, PBE, PBESOL, RPBE, WC, BLYP, LDA-C2, LDA-X, ZERO,"
              " HF, PBE0, B3LYP, HSE03, HSE06, EXX-X, HF-LDA, EXX, EXX-LDA,"
              " SHF, SX, SHF-LDA, SX-LDA, WDA, SEX, SEX-LDA]")
        exit(6)
        
    # check that the following args are not smaller than two
    for attr in ['fine_grid_step', 'ilevel']:
        if getattr(args, attr) < 2:
            print("[Err 05] Argument out of bounds: {} cannot be less than 2"
                  .format(attr))
            exit(5)

    # check that the following args are not smaller than one
    for attr in ['fine_grid_min', 'fine_grid_max', 'grid', 'max_scf_cycles',
                 'nodes', 'poll_wait', 'queue', 'tasks']:
        if getattr(args, attr) < 1:
            print("[Err 05] Argument out of bounds: {} cannot be less than 1"
                  .format(attr))
            exit(5)

    # check that the following args are not smaller than zero
    for attr in ['debug', 'extra_bands']:
        if getattr(args, attr) < 0:
            print("[Err 05] Argument out of bounds: {} cannot be less than 0"
                  .format(attr))
            exit(5)
    
    # check that the following args are not smaller than or equal to zero
    for attr in ['elec_enrg_tol', 'kpoint_spacing', 'mix_charge_amp',
                 'mix_spin_amp', 'volume_scale']:
        if getattr(args, attr) <= 0:
            print("[Err 05] Argument out of bounds: {} must be greater than 0"
                  .format(attr))
            exit(5)
     
    # check that the following args have correct relative values
    for attr_min, attr_max in [['fine_grid_min', 'fine_grid_max']]:
        if getattr(args, attr_min) > getattr(args, attr_max):
            print("[Err 05] Argument out of bounds: {} cannot be larger than {}"
                  .format(attr_min, attr_max))
            exit(5)

    return args

###############################################################################
#                                    MAIN                                     #
###############################################################################

if __name__ == '__main__':
    parser = setup_argparse()
    args = parse_args(parser)
    
    if args.load != "":     # if load file is specified, load data and plot it
        if debug_mode:
            print("[DEBUG] --load used, code will only load data and plot it.")
        plot_data(load_data(args.load), args.load, args.save)
        exit(0)             # exit, as this is all that was requested
        
    # set up clean environment every time running a CASTEP command
    clean_env = "module purge && module load {} &&".format(args.castep)
    if debug_mode:
        print("[DEBUG] clean_env command set to: "+clean_env)

    # set up fine grid as "step" values between (and including) "min" and "max"
    fine_grid = np.linspace(args.fine_grid_min, args.fine_grid_max,
                            args.fine_grid_step)
    if debug_mode:
        print("[DEBUG] generated fine_grid array: {}".format(fine_grid))
    
    if args.element == '':      # of element is not selected
        elements = listdir(args.cif_dir)    # select all elements
    else:               # otherwise create one item list from selected element
      elements = ['{}.cif'.format(args.element)]
    if debug_mode:
        print("[DEBUG] selected the following .cif files: {}".format(elements))

    print("**************************************************")
    print("*              == testgrids.py ==                *")
    print("**************************************************")
    print("* Job name: {:>36} *".format(args.name))
    print("* Element(s): {:>34} *".format("ALL" if len(elements) > 1
                                          else elements[0][:-4]))
    print("* Fine grid [start, stop, step]: {:>15} *".format("[{}, {}, {}]"
          .format(args.fine_grid_min, args.fine_grid_max, args.fine_grid_step)))
    print("* Type: {:>40} *".format("Primitive unit cell" if args.primitive
                                   else "Ground state structure"))
    print("* Debug level: {:>33} *".format(args.debug))
    print("**************************************************")
    print("* PARAM settings:                                *")
    print("*     BASIS_PRECISION: {:>25} *".format(args.basis_prec))
    print("*     CALCULATE_STRESS: {:>24} *"
          .format("True" if args.stress else "False"))
    print("*     ELEC_ENERGY_TOL: {:>22} eV *".format(args.elec_enrg_tol))
    print("*     GRID_SCALE: {:>30} *".format(args.grid))
    print("*     ILEVEL: {:>34} *".format(args.ilevel))
    print("*     MAX_SCF_CYCLES: {:>26} *".format(args.max_scf_cycles))
    print("*     MIXING_SCHEME: {:>27} *".format(args.mixing_scheme))
    print("*     MIX_CHARGE_AMP: {:>26} *".format(args.mix_charge_amp))
    print("*     MIX_SPIN_AMP: {:>28} *".format(args.mix_spin_amp))
    print("*     PERC_EXTRA_BANDS: {:>24} *".format(args.extra_bands))
    print("*     XC_FUNCTIONAL: {:>27} *".format(args.xc_functional))
    print("**************************************************")
    print("* CELL settings:                                 *")
    print("*     K-point spacing: {:>25} *".format(args.kpoint_spacing))
    print("*     Pseudopotential library: {:>17} *".format(args.pseudo_pot))
    print("*     Volume scale: {:>28} *".format(args.volume_scale))
    print("**************************************************")
    print("* SLURM settings:                                *")
    print("*     Nodes per job: {:>27} *".format(args.nodes))
    print("*     Poll time: {:>29} s *".format(args.poll_wait))
    print("*     Queue size: {:>25} jobs *".format(args.queue))
    print("*     Reservation string: {:>22} *"
          .format("None" if args.reserve == "" else args.reserve))
    print("*     Tasks per node: {:>26} *".format(args.tasks))
    print("**************************************************")

      
    if args.debug < 2:            # debug level 0-1 will allow for initial clean
        if args.reserve == "": cleanup_semaphores()  # cleanup unused semaphores
        cleanup_wkdir(args.name)  # clean up working directory
        
    # create simulation data and job files
    if args.debug < 3:          # debug levels 0-2 will allow generating data
        generate_job_data(elements, args.grid, fine_grid, args.name,
                          args.cif_dir, args.nodes, args.tasks, args.reserve,
                          args.stress, args.ilevel, args.pseudo_pot,
                          args.xc_functional, args.elec_enrg_tol,
                          args.basis_prec, args.extra_bands,
                          args.max_scf_cycles, args.mixing_scheme,
                          args.mix_charge_amp, args.mix_spin_amp,
                          args.kpoint_spacing, args.volume_scale)
    
    # submit jobs to SLURM and monitor their completion
    if args.debug < 4:          # debug levels 0-3 will allow for running jobs
        run_jobs(len(fine_grid)*7, args.queue, args.name, args.poll_wait)
    
    # after all simulations complete, collect, analyse and plot all data
    if args.debug < 5:          # debug levels 0-4 will allow analysing data
        plot_data(analyse_data(args.name, fine_grid, args.stress),
                  args.name, args.save)
    
    if args.reserve == "": cleanup_semaphores()  # cleanup any unused semaphores
    
    # if clean flag was used, clean up after simulation
    if args.clean:
        cleanup_wkdir(args.name)
        if debug_mode:
            print("[DEBUG] removed directory {}".format(args.name))
