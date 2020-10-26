import os
import json
import glob
import itertools

import numpy as np
from numpy import interp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


fig = plt.figure()
ax = fig.add_subplot(111)
for model in [
    'PCNN_ONE', 'PCNN_ATT'
]:
    dirs = glob.glob(f"/data4/tzhu/nyth_codes/bb_eval_models/422*/{model}/eval")
    all_data = list()
    for directory in dirs:
        with open(os.path.join(directory, "prc_opennre.json"), 'rt', encoding='utf-8') as fin:
            data = json.load(fin)
            all_data.append(data)

    all_recall = np.unique(np.concatenate([all_data[i]['recall'] for i in range(len(all_data))]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(len(all_data)):
        mean_precision += interp(all_recall, all_data[i]['recall'], all_data[i]['precision'])
    mean_precision /= float(len(all_data))

    ax.plot(all_recall, mean_precision, label="+".join(model.split('_')[:2]) + "(DSGT)", lw=2)


for model in [
    'PCNN_ONE', 'PCNN_ATT'
]:
    dirs = glob.glob(f"/data4/tzhu/nyth_codes/bb_eval_models/422*/{model}/eval")
    all_data = list()
    for directory in dirs:
        # for file in filter(lambda x: x.endswith('.json'), os.listdir(directory)):
        with open(os.path.join(directory, "b_prc_opennre.json"), 'rt', encoding='utf-8') as fin:
            data = json.load(fin)
            all_data.append(data)

    all_recall = np.unique(np.concatenate([all_data[i]['recall'] for i in range(len(all_data))]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(len(all_data)):
        mean_precision += interp(all_recall, all_data[i]['recall'], all_data[i]['precision'])
    mean_precision /= float(len(all_data))

    ax.plot(all_recall, mean_precision, label="+".join(model.split('_')[:2]) + "(MAGT)", lw=2)

for line, lst, lmk, cl in zip(ax.lines, ['-', ':', '--', '-.'], ['o', 'v', 's', 'D'], ['green', 'red', 'blue', 'black']):
    line.set_linestyle(lst)
    line.set_marker(lmk)
    line.set_markevery(100)
    line.set_fillstyle('none')
    line.set_color(cl)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([0.0, 0.4])
ax.set_ylim([0.7, 1.0])
ax.set_title('Precision-Recall')
ax.legend(loc='upper right')
ax.grid(True)
fig.savefig('prc_test_merge_correct_zoom_att_fix.png', format='png')
fig.savefig('prc_test_merge_correct_zoom_att_fix.pdf', format='pdf')


# -------------------------------------- k-covered-relation --------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
for model in [
    'PCNN_ONE', 'PCNN_ATT'
]:
    dirs = glob.glob(f"/data4/tzhu/nyth_codes/bb_eval_models/422*/{model}/eval")
    all_data = list()
    for directory in dirs:
        # for file in filter(lambda x: x.endswith('.json'), os.listdir(directory)):
        with open(os.path.join(directory, "k_covered_rel.json"), 'rt', encoding='utf-8') as fin:
            data = json.load(fin)
            all_data.append(data)
    ks = all_data[0]['all']['k']
    covered_rel = np.zeros_like(np.array(ks))
    for d in all_data:
        covered_rel += all_data[0]['all']['covered_rel']
    covered_rel = covered_rel / float(len(all_data))
    print(covered_rel[49])
    ax.plot(ks, covered_rel, label="+".join(model.split('_')[:2]), lw=4)

for line, lst, lmk, cl in zip(
        ax.lines, 
        itertools.cycle(['-', ':', '--', '-.']), 
        itertools.cycle(['o', 'v', 's', 'D']), 
        itertools.cycle(['green', 'red', 'blue', 'black'])):
    line.set_linestyle(lst)
    line.set_marker(lmk)
    line.set_markersize(10)
    line.set_markevery(500)
    line.set_fillstyle('none')
    line.set_color(cl)

ax.set_xlabel('k')
ax.set_ylabel('Number of Covered Relation')
ax.set_xlim([0, 2000])
ax.set_yticks(range(0, 22, 3))
ax.set_title('All Predictions')
ax.legend(loc='lower right')
ax.grid(True)
fig.savefig('k_covered_rel_all_att_fix.png', format='png')
fig.savefig('k_covered_rel_all_att_fix.pdf', format='pdf')


fig = plt.figure()
ax = fig.add_subplot(111)
for model in [
    'PCNN_ONE', 'PCNN_ATT'
]:
    dirs = glob.glob(f"/data4/tzhu/nyth_codes/bb_eval_models/422*/{model}/eval")
    all_data = list()
    for directory in dirs:
        # for file in filter(lambda x: x.endswith('.json'), os.listdir(directory)):
        with open(os.path.join(directory, "k_covered_rel.json"), 'rt', encoding='utf-8') as fin:
            data = json.load(fin)
            all_data.append(data)
    
    all_ks = np.unique(np.concatenate([all_data[i]['correct']['k'] for i in range(len(all_data))]))
    mean_covered_rel = np.zeros_like(all_ks)
    mean_covered_rel = mean_covered_rel.astype('float64')
    for i in range(len(all_data)):
        mean_covered_rel += interp(all_ks, all_data[i]['correct']['k'], all_data[i]['correct']['covered_rel'])
    mean_covered_rel /= float(len(all_data))

    ax.plot(all_ks, mean_covered_rel, label="+".join(model.split('_')[:2]), lw=4)

for line, lst, lmk, cl in zip(
        ax.lines, 
        itertools.cycle(['-', ':', '--', '-.']), 
        itertools.cycle(['o', 'v', 's', 'D']), 
        itertools.cycle(['green', 'red', 'blue', 'black'])):
    line.set_linestyle(lst)
    line.set_marker(lmk)
    line.set_markersize(10)
    line.set_markevery(500)
    line.set_fillstyle('none')
    line.set_color(cl)

ax.set_xlabel('Number of Correct Predictions')
ax.set_ylabel('Number of Covered Relation')
ax.set_xlim([0, 2000])
ax.set_yticks(range(0, 22, 3))
ax.set_title('Correct Predictions')
ax.legend(loc='lower right')
ax.grid(True)
fig.savefig('k_covered_rel_correct_att_fix.png', format='png')
fig.savefig('k_covered_rel_correct_att_fix.pdf', format='pdf')
