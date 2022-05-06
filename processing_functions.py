from helpers import create_new_folder, inputpath
import os
import fnmatch
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
from helpers import num_threads, play_filename, heuristic_func, do_multithread, optimal_path_heuristic, get_more_specific_df
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
def process_play(playpath_,playname,filename="results_plays", outpath="results/", heuristic='optimal', algorithm="astar_delaunay", speed_coefficient='optimal', save=False):
    csv = pd.read_csv(playpath_)
    try:
        if algorithm == "pitch_control":
            returner_pos, points, home, away, frame, pixels, pixel_values, frechet, yards, pitch_fraction, predicted_yards = pitch_control(csv)
            median_deviation = frechet
            mean_deviation = frechet
            d = {'play':[playname],'yardage':[yards],'median_deviation':[median_deviation],'mean_deviation':[mean_deviation], 'heuristic':[heuristic], 'algorithm':[algorithm], 'speed_coefficient':[speed_coefficient]}
        else:
            size,returner_pos,home,away,balls,lines, \
            times,optimal_paths,optimal_path_points,windows,\
            all_windows,optimal_points,play_direction,frechets,yardage_gained = process_frames(csv, True, True, heuristic=heuristic, algorithm=algorithm, speed_coefficient=speed_coefficient)
            median_deviation = np.median(np.array(frechets))
            mean_deviation = np.mean(np.array(frechets))
            d = {'play':[playname],'yardage':[yardage_gained],'median_deviation':[median_deviation],'mean_deviation':[mean_deviation], 'heuristic':[heuristic], 'algorithm':[algorithm], 'speed_coefficient':[speed_coefficient]}
        if save:
            df = pd.DataFrame.from_dict(d)
            df.to_csv(outpath+filename+".csv", mode='a')
    #If the play crashes just output nothing
    except Exception as e:
        d = {'play':[playname],'yardage':[None],'median_deviation':[None],'mean_deviation':[None], 'heuristic':[heuristic], 'algorithm':[algorithm], 'speed_coefficient':[speed_coefficient]}
        if save:
            df = pd.DataFrame.from_dict(d)
            df.to_csv(outpath+filename+".csv", mode='a')
    return d

def process_play_lam(play, inpath=inputpath+"/receiving_plays", filename="results_plays", outpath="results/", heuristic="optimal", algorithm="astar_delaunay", speed_coefficient="optimal", save=False):
    print(f"Processing play: {play}")
    filename_ = filename[:12] + f'_{play[:-4]}' + filename[13:]
    return process_play(inpath+"/"+play,play,filename_,outpath, heuristic=heuristic, algorithm=algorithm, speed_coefficient=speed_coefficient, save=save)

def process_all_plays(inpath=inputpath+"/receiving_plays", outpath="results/", heuristic='optimal', algorithm="astar_delaunay", speed_coefficient="optimal", num_procs=16, save=False):
    if multithread:
        pool = multiprocessing.Pool(num_procs)
        df = pd.DataFrame(columns=['play','yardage','median_deviation','mean_deviation', 'heuristic', 'algorithm', 'speed_coefficient'])
        results = pool.map(partial(process_play_lam, inpath=inpath+"/", filename=f"results_plays_heur={heuristic}_alg={algorithm}_scoeff={speed_coefficient}", outpath=outpath, heuristic=heuristic, algorithm=algorithm, speed_coefficient=speed_coefficient, save=save), os.listdir(inpath))
        for r in results:
            df = df.append(r, ignore_index=True)
        df.to_csv(f"results_plays_heur={heuristic}_alg={algorithm}_scoeff={speed_coefficient}.csv", mode='a', header=True)
    else:
        results = []
        for p in os.listdir(inpath):
            results.append(process_play_lam(p, inpath=inpath+"/", filename=f"results_plays_heur={heuristic}_alg={algorithm}_scoeff={speed_coefficient}", outpath=outpath, heuristic=heuristic, algorithm=algorithm, speed_coefficient=speed_coefficient, save=save))
        pd.concat(results).to_csv(f"results_plays_heur={heuristic}_alg={algorithm}_scoeff={speed_coefficient}.csv", header=True)

def aggregate_results(inpath="results", algorithm="astar", outpath="results"):	
    string_ = '*astar_s*'
    if algorithm == "astar":
        string_ = '*alg=astar_s*'
    elif algorithm == "astar_delaunay":
        string_ = '*alg=astar_delaunay*'
    elif algorithm == "pitch_control":
        string_ = '*alg=pitch_control*'
    files = fnmatch.filter(os.listdir(inpath), string_)
    yards = []
    devs = []
    other = open(f"{outpath}/results_full_{algorithm}.csv","a")
    other.write("play,yardage,median_deviation,mean_deviation,heuristic,algorithm,speed_coefficient\n")
    for f in files:
        row = open(f"{inpath}/"+f)
        lines = row.readlines()
        print(lines[1][2:])
        other.write(lines[1][2:])
    other.close()

def aggregate_results_pitch_control(inpath="data/", outpath="results"):	
    fractions = []
    yards_array = []
    predicted_yards_array = []
    frechets = []
    dataframe = []

    files = [f for f in os.listdir(f'{inpath}Receiving_Plays')]
    games_csv = pd.read_csv(f'{inpath}games.csv')
    # bad_files = [400, 448, 463]
    for string in files:
        csv = pd.read_csv(f'{inpath}Receiving_Plays/' + string)
        game = string.split('game')[1][:-4]
        game_row = get_more_specific_df(games_csv, 'gameId', int(game))
        home = game_row['homeTeamAbbr'].iloc[0]
        visitor = game_row['visitorTeamAbbr'].iloc[0]
        print(string, "file no", files.index(string), "/731")
        frechet, yards, pitch_fraction, predicted_yards, punt_returner = pitch_control(csv)
        returner_array = csv[csv['displayName'] == punt_returner]
        home_or_away = returner_array['team'].iloc[0]

        if home_or_away == 'home':
            returner_team = home
            other_team = visitor
        else:
            returner_team = visitor
            other_team = home
        if frechet != "Bad":
            frechets.append(frechet)
            yards_array.append(yards)
            predicted_yards_array.append(predicted_yards)
            fractions.append(pitch_fraction)
            array = [files.index(string), punt_returner, frechet, yards, predicted_yards, pitch_fraction, returner_team, other_team]
            dataframe.append(array)
    df = pd.DataFrame(dataframe, columns=['Game', 'Returner Name', 'Frechet Distance', 'Yards Gained', 'Predicted Yards', 'Control Fraction', 'Returning Team', 'Defending Team'])
    df.to_csv(f"{outpath}/results_full_pitch_control_detailed.csv")

def sort_wages(csv="results/ranking_stats", outpath="results"):
    csvp = pd.read_csv(f'{csv}.csv')
    wages = csv['Wage']
    print(min(wages))
    print(max(wages))
    wages_array = []
    for i in range(len(wages)):
        wages_ = (wages[i] - np.min(wages))/(np.max(wages) - np.min(wages))
        wages_array.append(wages_)
    df = pd.DataFrame(wages_array, columns=['wage'])
    df.to_csv(f'{outpath}/wages.csv')
    print(f"wages min: {min(wages)}, max: {max(wages)}")

def main(argv):
    logpath = None
    algorithm = "astar_delaunay"
    process_all = False
    play = play_filename[:-4]
    inpath = inputpath+"/receiving_plays"
    outputpath = 'results'
    heuristic = 'optimal'
    scoeff = 'optimal'
    num_procs = num_threads
    save = False
    try:
        opts, args = getopt.getopt(argv,"hap:o:i:l:m:n:q:k:s:S",["help","all","playid=","outpath=","inpath=","logpath=","mem=","num_procs=","algorithm=","heuristic=","speed_coeff="])
    except getopt.GetoptError:
        print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q <algorithm_type> -k <heuristic_type> -s <speed_coefficient_type> -S')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q <algorithm_type> -k <heuristic> -s <speed_coefficient_type> -S')
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
        elif opt in ("-n", "--num_procs"):
            num_procs = int(arg)
        elif opt in ("-q", "--algorithm"):
            if arg in {"astar", "astar_delaunay", "pitch_control"}:
                algorithm = arg
            else:
                raise ValueError(f'Error: argument -q / --algorithm must be one of "astar", "astar_delaunay", "pitch_control"')
        elif opt in ("-k", "--heuristic"):
            if arg in {"yardage", "euclidean", "custom", "optimal"}:
                heuristic = arg
            else:
                raise ValueError(f'Error: argument -k / --heuristic must be one of "yardage", "euclidean", "custom" or "optimal"')
        elif opt in ("-s", "--speed_coeff"):
            if arg in {"custom", "optimal"}:
                scoeff = arg
            else:
                raise ValueError(f'Error: argument -s / --speed_coeff must be "custom" or "optimal"')
        elif opt in ("-S", "--save"):
            save = True
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
        process_all_plays(inpath, outputpath+"/", heuristic=heuristic, algorithm=algorithm, speed_coefficient=scoeff, num_procs=num_procs, save=save)
        aggregate_results("results", "astar")
        aggregate_results("results", "astar_delaunay")
        aggregate_results("results", "pitch_control")
        aggregate_results_pitch_control(inpath="data/", outpath="results")
    else:
        process_play(inpath+"/"+play+".csv", play, outputpath, f"result_{play}", heuristic=heuristic, algorithm=algorithm, speed_coefficient=scoeff, save=save)
    if logpath is not None:
        print("Stopped printing to logfile")
        sys.stdout = old_stdout
        log_file.close()

if __name__ == '__main__':
    main(sys.argv[1:])