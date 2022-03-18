from vis_functions import*
import pandas as pd

# Change size of the figure
plt.rcParams['figure.figsize'] = [18,10]


df2 = pd.read_csv("csvs/Receiving_Plays/play143game2020111507.csv")
animate_one_play(df2)