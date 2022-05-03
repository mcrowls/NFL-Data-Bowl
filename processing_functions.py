from helpers import create_new_folder, inputpath
import os
import pandas as pd

import numpy as np
import random
import sys
not_unix = False
try:
    import resource
except:
    not_unix = True
from visualisation_functions import process_frames
from pitch_control import pitch_control
from helpers import num_threads, play_filename, heuristic_func, do_multithread, optimal_path_heuristic
import sys, getopt
from functools import partial
multithread = do_multithread
try:
    import multiprocessing
except:
    print("Error: Cannot import module 'multiprocessing' - defaulting to single-core")
    multithread = False

"""
Utils for RAM usage, CPU usage, Multithreading
"""
def memory_limit(percentage=0.8):
    if not_unix:
        raise ValueError("Error: memory_limit() can only be used on Ubuntu or WSL")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *percentage, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


"""
Get yardage and frechet distance for one play
"""
def process_play(playpath_,playname,filename="results_plays", outpath="results/", heuristic=optimal_path_heuristic, algorithm="astar_delaunay", save=False):
    csv = pd.read_csv(playpath_)
    try:
        if algorithm == "pitch_control":
            returner_pos, points, home, away, frame, pixels, pixel_values, frechet, yards, pitch_fraction, predicted_yards = pitch_control(csv)
        else:
            size,returner_pos,home,away,balls,lines, \
            times,optimal_paths,optimal_path_points,windows,\
            all_windows,optimal_points,play_direction,frechets,yardage_gained = process_frames(csv, True, True, heuristic=heuristic, algorithm=algorithm)
        median_deviation = np.median(np.array(frechets))
        mean_deviation = np.mean(np.array(frechets))
        if save:
            d = {'play':[playname],'yardage':[yardage_gained],'median_deviation':median_deviation,'mean_deviation':mean_deviation, 'algorithm':algorithm}
            df = pd.DataFrame(data=d)
            df.to_csv(outpath+filename+".csv", mode='a')
    #If the play crashes just output nothing
    except Exception as e:
        if save:
            d = {'play':[playname],'yardage':None,'median_deviation':None,'mean_deviation':None, 'algorithm':None}
            df = pd.DataFrame(data=d)
            df.to_csv(outpath+filename+".csv", mode='a')
    return df

def process_play_lam(play, inpath=inputpath+"/receiving_plays", filename="results_plays", outpath="results/", algorithm="astar_delaunay"):
    heuristic=optimal_path_heuristic
    if os.path.exists(outpath+filename+".csv"):
        results = pd.read_csv(outpath+filename+".csv")
        if play in results.values:
            return results.loc[results['filename']==play]
    print(f"Processing play: {play}")
    return process_play(inpath+"/"+play,play,filename,outpath, heuristic=heuristic, algorithm=algorithm)

def process_all_plays(inpath=inputpath+"/receiving_plays", outpath="results/", heuristic=lambda c, d: np.linalg.norm(c[0] - d[0]), algorithm="astar_delaunay"):
    if not os.path.exists(outpath):
        file = open(outpath,"w")
        file.write("id,play,yardage,median_deviation,mean_deviation,algorithm\n")
        file.close()
    if multithread:
        pool = multiprocessing.Pool(num_threads)
        results = pool.map(partial(process_play_lam, inpath=inpath+"/", filename="results_plays", outpath=outpath, algorithm=algorithm), os.listdir(inpath))
        df = pd.DataFrame(data=results)
        df.to_csv(f'results_plays_{algorithm}.csv', mode='a')
    else:
        #files = next(os.walk(outputpath_+f'physionet/data_{ecg_type}/{fn}/'))[1]
        #for file_ in next(os.walk(outputpath_+f'physionet/data_{ecg_type}/{fn}/'))[2]:
        #    if file_.endswith(f"{ecg_type}_signal.npy"):
        results = []
        for p in os.listdir(inpath):
            results.append(process_play_lam(p, inpath=inpath+"/", filename="results_plays", outpath=outpath, heuristic=heuristic, algorithm=algorithm))
        pd.concat(results).to_csv(f'results_plays_{algorithm}.csv')

def main(argv):
    logpath = None
    algorithm = "astar_delaunay"
    process_all = False
    play = play_filename[:-4]
    inpath = inputpath+"/receiving_plays"
    outputpath = 'results'
    try:
        opts, args = getopt.getopt(argv,"hap:o:i:l:m:q:",["help","all","playid=","outpath=","inpath=","logpath=","mem=","algorithm="])
    except getopt.GetoptError:
        print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q <algorithm_type>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q <algorithm_type>')
            sys.exit()
        elif opt in ("-a", "--all"):
            if opt in ("-p", "--playid"):
                print("Warning: --all option supplied; --playid is ignored")
            process_all = True
        elif opt in ("-p", "--playid"):
            if opt in ("-all", "--all"):
                print("Warning: --all option supplied; --playid is ignored")
            play = arg
        elif opt in ("-o", "--outpath"):
            outputpath = arg
        elif opt in ("-i", "--inpath"):
            inpath = arg
        elif opt in ("-l", "--logpath"):
            logpath = arg
        elif opt in ("-m", "--mem"):
            memory_limit(arg)
        elif opt in ("-q", "--algorithm"):
            if arg in {"astar", "astar_delaunay", "pitch_control"}:
                algorithm = arg
            else:
                raise ValueError(f'Error: argument -q / --algorithm must be one of "astar", "astar_delaunay", "pitch_control"')
    if logpath is not None:
        old_stdout = sys.stdout
        if logpath.endswith(".log"):
            print(f"Printing to logfile: {logpath}")
            log_file = open(logpath,"w")
        else:
            print(f"Printing to logfile: {logpath+'.log'}")
            log_file = open(logpath+".log","w")
        sys.stdout = log_file
    create_new_folder(outputpath)
    if process_all:
        process_all_plays(inpath, outputpath+"/", heuristic=heuristic_func, algorithm=algorithm)
    else:
        process_play(inpath+"/"+play+".csv", play, outputpath, f"result_{play}", heuristic=heuristic_func, algorithm=algorithm)
    if logpath is not None:
        print("Stopped printing to logfile")
        sys.stdout = old_stdout
        log_file.close()

if __name__ == '__main__':
    main(sys.argv[1:])
    