import numpy as np
import pandas as pd


ratings_csv = pd.read_csv('data/madden_21_ratings.csv')
names = ratings_csv['Full Name']
speeds = ratings_csv['Speed']
accelerations = ratings_csv['Acceleration']
agilities = ratings_csv['Agility']

mean_speed = np.mean(speeds)
std_speed = np.std(speeds)

zs = []
for speed in speeds:
    zs.append((speed - mean_speed)/std_speed)

avg_speed = 7

new_std = std_speed*(avg_speed/mean_speed)

df = pd.DataFrame(columns=['Name', 'Speed (yards/s)'],
    index=[j for j in range(np.size(speeds))])

for i in range(np.size(zs)):
    speed = zs[i]*new_std + avg_speed
    df.loc[i, :] = [names[i], speed]

print(df)

df.to_csv('data/Speed_Data.csv')
