import json

obj = json.load(open('data/gpt2_merged_predictions.json'))


scores1, scores2 = [], []
for element in obj['intrasentence']:
    id = element['id']
    score1 = element['score1']
    score2 = element['score2']

    scores1.append(score1)
    scores2.append(score2)

print(f'Average score for data 1: {sum(scores1) / len(scores1) * 10000}')
# 65.97741972437179

print(f'Average score for data 2: {sum(scores2) / len(scores2) * 10000}')
# 71.97626358386856
