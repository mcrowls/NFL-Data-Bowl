# NFL-Data-Bowl

Coursework for the Applied Data Science unit analysing, inferring and visualising the NFL Big Data Bowl 2022 Comptetition data.

The NFL Big Data Bowl 2022 data is here [https://www.kaggle.com/c/nfl-big-data-bowl-2022], and the Madden 2021 data is here [https://www.kaggle.com/datasets/dtrade84/madden-21-player-ratings]. This project aims to analyse Punt Returns, specifically the
Punt Returner's path, the method used to calculate this, and how it can be optimised. 
We use three methods to create an optimal path for the Punt Returner; Pitch Control, and with the Delaunay Triangulations classic A* Search and Mod(ified) A*.

### The repo contains the following notebooks:

* NFL_Data_Cleaning: A notebook running through the data cleaning process for the NFL Big Data Bowl 2022 data
* NFL_Data_Visualise: A notebook containing scripts to save a play as an .mp4 or .html file, alternatively use 'visualisation_functions.py'
* NFL_Data_Results: A notebook showcasing the results in the paper and the comparisons of the techniques used for optimal path calculation

### And the following scripts, with the options commented above

```py
# --all / -a: processes all punt returns (plays)  
# OR 
# --playid / -p: id of the play to process (e.g. play2365-game2020110108)
#
# --inpath / -i: path to the folder containing the play csv's   
# --outpath / -o : path to store the results (/results or /visualisations)
# --logpath / -l : path to store the output - print statements from the logger
# --mem / -m : proportion (% / 100) in range [0, 1] of system memory available during processing (default=0.8=80%)
# --algorithm / -q : which model to use to calculate optimal path; 
#                       - astar_delaunay: the project's Delaunay window implementation of A* (default, False) 
#                       - astar: the standard A* pathfinding algorithm (True)
#                       - pitch_control: the Pitch Control method as described in the paper
processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path> -l <logfile_full_filepath> -m <percentage_of_mem_usage_allowed> -q <algorithm_type>



# --playid / -p: id of the play to process (e.g. play2365-game2020110108)
# --inpath / -i: path to the folder containing the play csv's   
# --outpath / -o: path to store the results (/results or /visualisations)
# --visfunc / -v: function to use to create the play visualisation; new/old/funcanim
# --algorithm / -q : which model to use to calculate optimal path; 
#                       - astar_delaunay: the project's Delaunay window implementation of A* (default, False) 
#                       - astar: the standard A* pathfinding algorithm (True)
#                       - pitch_control: the Pitch Control method as described in the paper
visualisation_functions.py -p <play_id> -i <input_path> -o <output_path> -v <"new"/"old"/"funcanim"> -q <algorithm_type>
```

## Optimal Path for Frame 3 from play116-game2021010301 using A*
![Optimal Path for Frame 3 from play116-game2021010301 using A*](./res/-p%20play116-game2021010301%20-v%20new%20-q%20astar.png)
## Optimal Path for Frame 3 from play116-game2021010301 using ModA* & Delaunay Triangulation
![Optimal Path for Frame 3 from play116-game2021010301 using ModA* & Delaunay Triangulation](./res/-p%20play116-game2021010301%20-v%20new%20-q%20astar_delaunay.png)

## Optimal Path for play116-game2021010301 using Pitch Control
![Optimal Path for play116-game2021010301 using Pitch Control](./res/-p%20play116-game2021010301%20-v%20new%20-q%20pitch_control.png)



### Global variables, input/output paths and defaults for command-line options can be changed in helpers.py:

```py
"""# Global Variables / Paths"""

"""## Edit These"""
input_folderpath = "data"
output_folderpath = "data"
play_filename = 'play2365-game2020110108.csv'
vis_output_folderpath = "visualisations"
results_output_folderpath = "results"
drive_folderpath = "Colab Notebooks"
useDrive = False
avg_player_speed = 7
num_threads = 16

heuristic_func = lambda c, d: np.linalg.norm(c[0] - d[0])
```

### There are also the following commands which setup the environment (dependencies, notebook options):

```py
pip install -U tk
pip install -U jupyter_console
pip install -U ipywidgets
pip install -U ffmpeg-python
jupyter nbextension enable --py widgetsnbextension
```

### There is a conda environment which can be activated via environment.yml; use the following commands to setup the environment:

```py
conda env create --name <env_name> --file=environments.yml
```