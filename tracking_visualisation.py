from vis_functions import*
import pandas as pd



# Change size of the figure
plt.rcParams['figure.figsize'] = [18, 10]
csv = pd.read_csv('csvs/Receiving_Plays/play116-game2021010301.csv')
animate_return(csv)







