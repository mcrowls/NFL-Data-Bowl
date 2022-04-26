# NFL-Data-Bowl

Coursework for the Applied Data Science unit analysing, inferring and visualising the NFL Big Data Bowl 2022 Comptetition data.

The NFL Big Data Bowl 2022 data is here [], and the Madden 2021 data is here []. This project aims to analyse Punt Returns, specifically the
Punt Returner's path, the method used to calculate this, and how it can be optimised.

### The repo contains the following notebooks:

* NFL_Data_Cleaning: A notebook running through the data cleaning process for the NFL Big Data Bowl 2022 data
* NFL_Data_Visualise: A notebook containing scripts to save a play as an .mp4 or .html file, alternatively use 'visualisation_functions.py'
* NFL_Data_Results: A notebook showcasing the results in the paper and the comparisons of the techniques used for optimal path calculation

### And the following scripts, with the options commented above

```py
# --all: processes all punt returns (plays)  OR --playid: id of the play to process (e.g. play2365-game2020110108)
# --inpath: path to the folder containing the play csv's   --outpath: path to store the results (/results or /visualisations)
processing_functions.py [-a | -p <play_id>] -i <input_path> -o <output_path>
# --all: processes all punt returns (plays)   --playid: id of the play to process (e.g. play2365-game2020110108)
# --inpath: path to the folder containing the play csv's   --outpath: path to store the results (/results or /visualisations)
# --visfunc: function to use to create the play visualisation; new/old/funcanim
visualisation_functions.py -p <play_id> -i <input_path> -o <output_path> -v <"new"/"old"/"funcanim">
```

There are also the following dependancies:

```py
pip install -U tk
pip install -U jupyter_console
pip install -U ipywidgets
pip install -U ffmpeg-python
jupyter nbextension enable --py widgetsnbextension
```