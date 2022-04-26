from helpers import create_new_folder, inputpath
import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import random
import sys
import resource
from visualisation_functions import process_frames
from helpers import num_threads, play_folderpath
import sys, getopt
from functools import partial

"""
Utils for RAM usage, CPU usage, Multithreading
"""
def memory_limit(percentage=0.8):
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
def process_play(playpath_,playname,outpath="results/",filename="results_plays"):
    csv = pd.read_csv(playpath_)
    try:
        size,returner_pos,home,away,balls,lines, \
        times,optimal_paths,optimal_path_points,windows,\
        all_windows,optimal_points,play_direction,frechets,yardage_gained = process_frames(csv, True, True)
        median_deviation = np.median(np.array(frechets))
        mean_deviation = np.mean(np.array(frechets))
        d = {'play':[playname],'yardage':[yardage_gained],'median_deviation':median_deviation,'mean_deviation':mean_deviation}
        df = pd.DataFrame(data=d)
        df.to_csv(outpath+filename+".csv", mode='a', header=False)
    #If the play crashes just output nothing
    except Exception as e:
        d = {'play':[playname],'yardage':None,'median_deviation':None,'mean_deviation':None}
        df = pd.DataFrame(data=d)
        df.to_csv(outpath+filename+".csv", mode='a', header=False)

def process_play_lam(play, inpath=inputpath+"/receiving_plays", outpath="results/", filename="results_plays"):
    results = pd.read_csv(outpath+filename+".csv")
    if play in results.values:
        return play.loc[play['play']==play]
    print(f"Processing {play['play'].values[0]}: {play}")
    process_play(inpath+"/"+play,play,outpath,filename)

def process_all_plays(inpath=inputpath+"/receiving_plays", outpath="results/results_plays.csv"):
    if not os.path.exists(outpath):
        file = open(outpath,"w")
        file.write("id,play,yardage,median_deviation,mean_deviation\n")
        file.close()
    pool = Pool(num_threads)
    pool.map(partial(process_play_lam, filename="results_plays", inpath=inpath), os.listdir(inpath))

def main(argv):
    log = False
    process_all = False
    play = play_folderpath[:-4]
    inpath = inputpath+"/receiving_plays"
    outputpath = 'results'
    try:
        opts, args = getopt.getopt(argv,"hap:o:i:l",["help","all","playid=","outpath=","inpath=","log"])
    except getopt.GetoptError:
        print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l')
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
        elif opt in ("-l", "--log"):
            log = True
    create_new_folder(outputpath)
    if process_all:
        process_all_plays(inpath, outputpath+"/results_plays.csv")
    else:
        process_play(inpath+"/"+play+".csv", play, outputpath, f"result_{play}")

if __name__ == '__main__':
    get_memory(0.8)
    main(sys.argv[1:])
    