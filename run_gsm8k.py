import os
import json
import numpy as np
import logging
from datasets import load_dataset
from tqdm import tqdm
from llms import CLM, OpenAIModel



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



if __name__ == "__main__":

    total_tokens = 0
    # load the dataset from json file
    dataset = []
    with open("gsm_8k_rc.json") as f:
        dataset = json.load(f)
    fprompt = open("prompt.txt", "r")
    base_prompt = fprompt.read().strip()
    fprompt.close()
    fp = open("gsm_8k_rc_llama.json", "a")
    # vicuna = CLM(model_name="vicuna-7b-v1.5", model_dir="lmsys/vicuna-7b-v1.5", cache_file="vicuna.cache")
    # mpt = CLM(model_name="mpt-7b", model_dir="mosaicml/mpt-7b-instruct", cache_file="mpt.cache")
    # chatgpt = OpenAIModel(model_name='ChatGPT', cache_file='turbo_cache.json')
    llama_64b = CLM(model_name="llama-70b", model_dir="meta-llama/Llama-2-13b-chat-hf", cache_file="llama.cache")
    for i in tqdm(dataset):
        question = i["question"]
        prompt = f"{base_prompt}\nQuestion: {question}\nLets think step by step"

        pred = llama_64b.generate(prompt, max_sequence_length=2048, max_output_length=512)

        generated_answer = pred[0]
        # total_tokens += pred[1]['usage']['total_tokens']
        i['llama70b_answer'] = generated_answer
    
        json.dump(i , fp)
        fp.write("\n")
    fp.close()
    print_cost_estimates(total_tokens, "GSM8k", "gpt-3.5-turbo")