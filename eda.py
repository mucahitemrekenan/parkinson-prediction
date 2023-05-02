import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
from tqdm import tqdm
from sklearnex  import patch_sklearn
patch_sklearn()
matplotlib.use('Qt5Agg')


defog_path = 'data/train/defog/'
defog_files = os.listdir(defog_path)
defog_data = pd.DataFrame()

for file in tqdm(defog_files):
    patient_defog_data = pd.read_csv(defog_path+file)
    patient_defog_data['session_id'] = file.replace('.csv', '')
    defog_data = pd.concat([defog_data, patient_defog_data])

# we yield length of data through all patients as minute
defog_sampling_rate = 100
defog_session_times = defog_data.value_counts(subset='session_id').reset_index()
defog_session_times['length_min'] = defog_session_times['count'] / (defog_sampling_rate * 60)

# plt.plot('session_id', 'length_min', data=defog_session_times)
# plt.xticks(rotation=90)
# plt.show()
# plt.clf()

# for plotting only this yields the same plot practically
# plt.plot(defog_data['session_id'].value_counts() / (100 * 60))
# plt.xticks(rotation=90)
# plt.show()
# plt.clf()

# for column in ['StartHesitation', 'Turn', 'Walking']:
#     print(defog_data.value_counts(subset=column, normalize=True))

# we inspect target variables by session_id which sesssion_id does which movement
# and how much
# move_counts = defog_data.value_counts(subset=['StartHesitation', 'Turn', 
#                                        'Walking', 'session_id']).reset_index()

# hesitation_counts = move_counts.loc[move_counts['StartHesitation'] == 1]
# turn_counts = move_counts.loc[move_counts['Turn'] == 1]
# walking_counts = move_counts.loc[move_counts['Walking'] == 1]

# plt.plot('session_id', 'count', '.', data=hesitation_counts)
# plt.plot('session_id', 'count', '.', data=turn_counts)
# plt.plot('session_id', 'count', '.', data=walking_counts)
# plt.legend(['StartHesitation', 'Turn', 'Walking'])
# plt.xticks(rotation=90)
# plt.grid(visible=True, axis='both', which='major')
# plt.show()
# plt.clf()

sessions = defog_data['session_id'].unique()
sensor_data = defog_data[defog_data['Valid'] == True].copy()

for number, session in tqdm(enumerate(sessions[:5])):
    sensor_data.loc[sensor_data['session_id'] == session]\
        .plot(x='Time', y=['Event','AccV', 'AccML', 'AccAP'], figsize=(19.1,9.6),
              legend=True)
plt.show()

for number, session in tqdm(enumerate(sessions[10:20])):
    session_data = sensor_data.loc[sensor_data['session_id'] == session,
                                   ['Event', 'AccV', 'AccML', 'AccAP']].copy()
    # plt.figure(number, figsize=(19.1,9.6), tight_layout=True)
    # plt.plot(session_data['AccV'], '-')
    # plt.plot(session_data['StartHesitation'], 'x')
    # plt.plot(session_data['Turn'], '.')
    # plt.plot(session_data['Walking'], '|')
    # plt.legend(['AccV', 'StartHesitation', 'Turn', 'Walking'])

    x = session_data.reset_index(drop=True).index
    y1 = session_data['AccV']
    y2 = session_data['AccV']
    y3 = session_data['AccML']
    y4 = session_data['AccAP']

    fig, ax1 = plt.subplots(figsize=(19.1,9.6), tight_layout=True)
    fig.suptitle(f'{number}_{session}_StartHesitation')
    color = 'tab:blue'
    ax1.set_ylabel('AccV', color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('StartHesitation', color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # ------------------------------------------
    fig2, ax3 = plt.subplots(figsize=(19.1,9.6), tight_layout=True)
    fig2.suptitle(f'{number}_{session}_Turn')
    color = 'tab:blue'
    ax3.set_ylabel('AccV', color=color)
    ax3.plot(x, y1, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()
    color = 'tab:red'
    ax4.set_ylabel('Turn', color=color)
    ax4.plot(x, y3, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    
    # ------------------------------------------
    fig3, ax5 = plt.subplots(figsize=(19.1,9.6), tight_layout=True)
    fig3.suptitle(f'{number}_{session}_Walking')
    color = 'tab:blue'
    ax5.set_ylabel('AccV', color=color)
    ax5.plot(x, y1, color=color)
    ax5.tick_params(axis='y', labelcolor=color)

    ax6 = ax5.twinx()
    color = 'tab:red'
    ax6.set_ylabel('Walking', color=color)
    ax6.plot(x, y4, color=color)
    ax6.tick_params(axis='y', labelcolor=color)


plt.show()
plt.clf()


metadata = pd.read_csv('data/tdcsfog_metadata.csv')
events = pd.read_csv('data/events.csv')
tasks = pd.read_csv('data/tasks.csv')

session = pd.read_csv('data/train/defog/f9efef91fb.csv')

plt.figure(figsize=(19.1,9.6), tight_layout=True)
plt.plot(session['AccV'], '-')
plt.plot(session['StartHesitation'], 'x')
plt.plot(session['Turn'], '.')
plt.plot(session['Walking'], '|')
plt.legend(['AccV', 'StartHesitation', 'Turn', 'Walking'])
plt.show()

# sns.heatmap(defog_data.corr(), cmap='coolwarm', annot=True)
