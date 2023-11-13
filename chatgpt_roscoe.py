import sys
import time
import os
import json
import logging
import pickle
from tqdm import tqdm
from llms import OpenAIModel
from scipy.stats import somersd, pearsonr
from sklearn.metrics import confusion_matrix

def print_cost_estimates(total_tokens, task, model):
    # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    # Number of tokens are roughly 4/3 of the number of words
    # https://openai.com/pricing
    # if we use davinci-003, the cost is $0.02 per 1000 tokens
    # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
    if model == "davinci-003":
        rate = 0.02
    elif model == "gpt-3.5-turbo":
        rate = 0.002

    total_cost = total_tokens * rate / 1000

    # print the total words, tokens, and cost along with rate
    print("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_tokens, total_tokens))


def somerd_correlation(prediction, gt):
    table = confusion_matrix(gt, prediction)
    res = somersd(table)
    return res.statistic, res.pvalue

def pearson_correlation(prediction, gt):
    res = pearsonr(prediction, gt)
    return res.statistic, res.pvalue


if __name__ == "__main__":
    total_tokens = 0
    with open("chatgpt_metric_prompt_chain.txt", "r") as f:
        base_prompt = f.read().strip()

    with open("gsm_8k_rc.json", 'r') as f:
        dataset = json.load(f)

    fp = open("gsm_8k_rc_new_chatgpt_v3.json", "w")
    # dataset = []
    # for line in fp:
    #     dataset.append(json.loads(line))
 
    # for i in tqdm(data):
    #     answer = i['llama70b_answer']
    #     try:
    #         index = answer.index('Question')
    #     except:
    #         index = len(answer)
    #     new_answer = i['llama70b_answer'][:index].strip()
    #     i['llama70b_answer'] = new_answer

    # fwrite = open("gsm_8k_rc_llama2.json", "w")
    # json.dump(data, fwrite, indent=4)


    chatgpt = OpenAIModel(model_name='ChatGPT', cache_file='turbo_cache.json')
    for i in tqdm(dataset):
        question = i["question"]
        full_solution = [j['step_text'] for j in i['step_metrics']]
        # for step in range(len(i['step_metrics'])):
        # solution = " ".join(full_solution[:step]).strip()
        # next_step = full_solution[step].strip()
        # prompt = f"{base_prompt}\nQuestion: {question}\nSolution: {solution}\nNext Step: {next_step}\nOutput:"
        prompt = f"{base_prompt}\nQuestion: {question}\nSolution: {' '.join(full_solution)}\nOutput:"
        pred = chatgpt.generate(prompt, max_sequence_length=2048, max_output_length=5)
        time.sleep(0.5)
        generated_answer = pred[0]
        total_tokens += pred[1]['usage']['total_tokens']
        # i['step_metrics'][step]['chatgpt_answer'] = generated_answer
        i['chatgpt_answer'] = generated_answer
        json.dump(i , fp)
        fp.write("\n")

    # print_cost_estimates(total_tokens, "GSM8k", "gpt-3.5-turbo")

    gt_mapping = {'yes': 1, 'no': 0} # GT is error
    pred_mapping = {'correct': 0, 'incorrect': 1}

    # pred, gt = [] , []
    # for i in dataset:
    #     gt.append(gt_mapping[i['overall_metrics']['contradiction']])
    #     if sum([pred_mapping[s['chatgpt_answer']] for s in i['step_metrics']]) > 1:
    #         pred.append(1)
    #     else:
    #         pred.append(0)

    # print(f'Overall {somerd_correlation(pred, gt)}')

    dimesnions = list(dataset[0]['step_metrics'][0].keys())
    # delete the following dimensions: step_text
    print(dimesnions)
    dimesnions.remove('step_text')
    dimesnions.remove('step')
    # dimesnions.remove('chatgpt_answer')
    # print(dimesnions)
    # ['contradiction', 'grammar', 'logical_deduction', 'final_answer_wrong','extra_useless_info','intermediate_factual_info', 'droppable_step', 'world_knowledge', 'math_error']
    # for dim in dimesnions:
    #     pred, gt = [], []
    #     for i in dataset:
    #         for s in i['step_metrics']:
    #             if s[dim] == "" or s['chatgpt_answer'].lower() in ["invalid", "<invalid>"]:
    #                 # print(s)
    #                 continue
    #             gt.append(gt_mapping[s[dim]])
    #             pred.append(pred_mapping[s['chatgpt_answer']])
    #     # print(f'Pearson Correlation for {dim} {pearson_correlation(pred, gt)}')                
    #     print(f'{dim} {somerd_correlation(pred, gt)}')

    pred, gt = [], {}
    for indx, i in enumerate(dataset):
        """
        i = {"model": "flant5", "metadata_example_idx": 0, "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "gold": "Step 1 - Janet eats 3 duck eggs for breakfast and bakes 4 into muffins so 3 + 4 = <<3+4=7>>7 duck eggs are used. Step 2 - Each day Janet's ducks lay 16 eggs and she uses 7, 16 - 7 = <<16-7=9>>9 duck eggs are for sale. Step 3 - She sells her eggs for $2 per egg and has 9 available for sale so 2 * 9 = $<<2*9=18>>18 per day. Step 4 - A: 18", "cot": "Solution: Janet eats 3 eggs for breakfast and 4 for muffins so she has 3 + 4 = 7 eggs. She sells the remaining eggs at the farmers' market for $2 per egg so she makes 2 * 7 = $14. She makes $14 every day at the farmers' market. Answer: 14.", "step_metrics": [{"step_text": "Janet eats 3 eggs for breakfast and 4 for muffins so she has 3 + 4 = 7 eggs.", "Grammar": "no", "Coherency": "no", "Hallucination": "no", "Missing step": "no", "Repetition": "no", "Factuality": "no", "Redundancy": "no", "Commonsense": "yes", "Arithmetic": "no", "step": 1}, {"step_text": "She sells the remaining eggs at the farmers' market for $2 per egg so she makes 2 * 7 = $14.", "Grammar": "no", "Coherency": "no", "Hallucination": "no", "Missing step": "no", "Repetition": "no", "Factuality": "no", "Redundancy": "no", "Commonsense": "no", "Arithmetic": "no", "step": 2}, {"step_text": "She makes $14 every day at the farmers' market.", "Grammar": "no", "Coherency": "no", "Hallucination": "no", "Missing step": "no", "Repetition": "yes", "Factuality": "no", "Redundancy": "no", "Commonsense": "no", "Arithmetic": "no", "step": 3}, {"step_text": "Answer: 14.", "Grammar": "no", "Coherency": "no", "Hallucination": "no", "Missing step": "no", "Repetition": "no", "Factuality": "no", "Redundancy": "no", "Commonsense": "no", "Arithmetic": "no", "step": 4}], "chatgpt_answer": "correct"}
        """
        pred.append(pred_mapping[i['chatgpt_answer']])
        gt[indx] = {}
        for dim in dimesnions:
            ans = 0
            for s in i['step_metrics']:
                if s[dim] == "":
                    continue
                ans += gt_mapping[s[dim]]
                
            gt[indx][dim] = int(ans>0)

    # assert len(pred) == len(gt)
    # print(f'Pearson Correlation for {dim} {pearson_correlation(pred, gt)}')  

    # invert 2 dimensional dictionary gt


    for key in dimesnions:
        print(key)
        gt_dim = [gt[i][key] for i in range(len(dataset))]
        print(somerd_correlation(pred, gt_dim))       
    # print(f'{dim} {somerd_correlation(pred, gt)}')


