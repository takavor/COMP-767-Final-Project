import json
import matplotlib.pyplot as plt
import numpy as np


# load data
with open('data/llama_preliminary_1.json') as file_1:
    data_1 = json.load(file_1)

with open('data/llama_preliminary_2.json') as file_2:
    data_2 = json.load(file_2)


from metrics import get_metrics
lms_1, ss_1, icat_1 = get_metrics(data_1)
lms_2, ss_2, icat_2 = get_metrics(data_2)

print('Data 1')
print(f'LMS: {lms_1}')
print(f'SS: {ss_1}')
print(f'iCAT: {icat_1}')
print('---')
print('Data 2')
print(f'LMS: {lms_2}')
print(f'SS: {ss_2}')
print(f'iCAT: {icat_2}')


def graph2():
    # Scores for GPT-2 and LLAMA 2 on Data 1 and Data 2
    gpt2_scores = [65.97741972437179, icat_1]
    llama2_scores = [71.97626358386856, icat_2]

    # The x locations for the groups
    x = np.arange(len(gpt2_scores))
    # The width of the bars
    width = 0.35

    fig, ax = plt.subplots()

    # Plotting the bars for GPT-2 and LLAMA 2
    bars1 = ax.bar(x - width/2, gpt2_scores, width, label='Data 1')
    bars2 = ax.bar(x + width/2, llama2_scores, width, label='Data 2')

    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.set_ylabel('iCAT Scores')
    ax.set_title('iCAT Score Comparison by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(['GPT-2', 'LLAMA 2'])

    # Placing the legend outside the plot area, on the top right
    ax.legend(title='Dataset', loc='upper right', bbox_to_anchor=(1, 1))

    # Labeling the bars with their respective iCAT scores
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    # plt.show()
    plt.savefig('results/icat_scores.png')


def graph():

    # Scores for Data 1 and Data 2 as provided in the uploaded image
    scores_data_1 = {'LMS': lms_1, 'SS': ss_1, 'iCAT': icat_1}
    scores_data_2 = {'LMS': lms_2, 'SS': ss_2, 'iCAT': icat_2}

    labels = list(scores_data_1.keys())
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots()

    # Generate bars for Data 1
    rects1 = ax.bar(x - width/2, scores_data_1.values(), width, label='Data 1')

    # Generate bars for Data 2
    rects2 = ax.bar(x + width/2, scores_data_2.values(), width, label='Data 2')

    ax.set_xlabel('Score Type')
    ax.set_ylabel('Score Value')
    ax.set_title('Scores for Llama2 - Data 1 vs Data 2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='Dataset', loc='upper right', bbox_to_anchor=(1, 1))

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('results/llama2_scores.png')


graph()
graph2()
