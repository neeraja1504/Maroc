import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy

def get_model_results(questions, prompt, model_name="flan-t5-large", device="cuda:0"):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")
    responses = {}
    for k, v in questions.items():
        print(f"{k+1}/200", end='\r')
        q_prompt = prompt.format(question=v)
        inputs = tokenizer(q_prompt, max_length=512, return_tensors="pt").to(device)
        outputs = model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=2048)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses[k] = answer
    return responses

if __name__ == '__main__':
    with open("./gsm_8k.json") as f:
        gsm_8k_data = json.load(f)

    questions = {q["index"]:q["question"] for q in gsm_8k_data}

    with open("./prompt.txt") as f:
        prompt = f.read()
    
    model_name = "flan-t5-large",
    responses = get_model_results(questions, prompt, model_name=model_name, device="cuda:0")

    model_outputs = []
    for item in gsm_8k_data:
        response_outputs = deepcopy(item)
        response_outputs[f"{model_name}_output"] = responses[item["index"]]
        model_outputs.append(response_outputs)
        
    with open(f"./gsm_8k_{model_name}.json", "w+") as f:
        json.dump(model_outputs, f, indent=4)
