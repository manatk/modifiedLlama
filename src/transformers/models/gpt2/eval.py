import torch
from transformers import AutoTokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel
from custom_model import GPT2WithThresholdedAttention
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate import load
import argparse

MAX_LENGTH = 1000 # Max length for input sequence

# Adding argparse for choosing mode
parser = argparse.ArgumentParser(description='Select mode.')
parser.add_argument('mode', metavar='N', type=str, nargs='?', default = 'r',
                    help='Choose whether to run model with all alpha values (\'a\'), one alpha value (\'r\'), or plot (\'p\')')
parser.add_argument('alpha', type = float, nargs='?', default = 1,
                    help='choose alpha value')
parser.add_argument('task', type = str, nargs='?', default = "BookSum",
                    help='choose benchmark task')
args = parser.parse_args()

perplexity = load("perplexity", module_type="metric")

'''
Creates attention mask from attentions = Tuple of tensor (batch_size, num_heads, seq_length, seq_length)

Attention mask is of size (batch_size, seq_length)
'''
def create_attention_mask(attentions, alpha):
    attentions = attentions[0]
    print("Creating attention masks...")
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=(2,3)) # sum over heads and sequence_length rows
    threshold = np.percentile(sums, (1 - alpha) * 100) # Get ((1-alpha) * 100)th percentile
    masks = np.where(sums < threshold, 0, 1)
    print("Attention masks created.")
    return torch.from_numpy(masks).to(attentions[0].device)


'''
Runs different alpha_values to generate various models using the mask method, with parameters specified by config.
Gets benchmark scores of each model. Then saves generated plots on save_path.
'''
def evaluate_model_mask_alpha(device, alpha, config, tokenizer, task, n, write_path, layer=0):
    testData = None
    if task == "BookSum":
        testData = load_dataset("kmfoda/booksum")["test"]
    elif task == "LegalBench":
        testData = load_dataset("nguha/legalbench", "consumer_contracts_qa")["test"]
    generated_texts = []
    for i in range(0, n):
        print("Forward passing for alpha = " + str(alpha), ", iteration " + str(i+1) + "/" + str(n))
        inputs = None
        if task == "BookSum":
            prompt = testData[i]['chapter'] + "Summarize this chapter"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        elif task == "LegalBench":
            prompt = testData[i]['contract'] + testData[i]["question"]
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        config.alpha = alpha
        generated_text = forward_pass(device, config, inputs, tokenizer)
        generated_texts.append(generated_text)
    # Get score
    bookSumScore = get_score(generated_texts)
    data = []
    data.append({"alpha": alpha, "task": task, "score": bookSumScore, "n": n})
    #data.append({"alpha": alpha, "task": "LegalBench", "score": legalBenchScore})
    df = pd.DataFrame(data)
    df.to_csv(write_path, mode = 'a', index=False)

'''
Runs different alpha_values to generate various models using the mask method, with parameters specified by config.
Gets benchmark scores of each model. Then saves generated plots on save_path.

Tests on the first n examlpes in the benchmark task
'''
def evaluate_model_mask(device, config, tokenizer, task, n, write_path, layer=0):
    testData = None
    if task == "BookSum":
        testData = load_dataset("kmfoda/booksum")["test"]
    elif task == "LegalBench":
        testData = load_dataset("nguha/legalbench", "consumer_contracts_qa")["test"]
    alpha_values = [0, .5, .6, .7, .8, .9, .95, 0.97, 0.99]
    scores = {"BookSum": {alpha: [] for alpha in alpha_values}, "LegalBench": {alpha: [] for alpha in alpha_values}}
    # Create model for each alpha value
    for alpha in alpha_values:
        generated_texts = []
        for i in range(0, n):
            print("Forward passing for alpha = " + str(alpha), ", iteration " + str(i+1) + "/" + str(n))
            inputs = None
            if task == "BookSum":
                prompt = testData[i]['chapter'] + "Summarize this chapter"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            elif task == "LegalBench":
                prompt = testData[i]['contract'] + testData[i]["question"]
                inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            config.alpha = alpha
            generated_text = forward_pass(device, config, inputs, tokenizer)
            generated_texts.append(generated_text)
        # Get score
        score = get_score(generated_texts)
        data = []
        data.append({"alpha": alpha, "task": task, "score": score, "n": n})
        #data.append({"alpha": alpha, "task": "LegalBench", "score": legalBenchScore})
        df = pd.DataFrame(data)
        df.to_csv(write_path, mode = 'a', index=False)

'''
Forward pass through model through mask method

Example (x,y) to pass is specified hrough testData and index i
Mdoel is specified through config
We use alpha to create attentionmask
'''
def forward_pass(device, config, inputs, tokenizer):
    with torch.no_grad():
        model = GPT2WithThresholdedAttention.from_pretrained('gpt2', config=config).to(device)
        output = model.generate(inputs.input_ids.to(device), max_length = MAX_LENGTH+1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


'''
Plots values from write_path, and saves plot to save_path
'''
def plot_alpha_summation_benchmarks_from_csv(load_path, save_path, n):
    # Load the data
    data = pd.read_csv(load_path)
    plt.figure()
    alpha = data['alpha']
    scores = data['score']
    tasks = data['task']

    # Step 3: Create scatter plot
    plt.figure(figsize=(10, 6))

    # Get unique tasks
    unique_tasks = data['task'].unique()

    # Plot each task with different colors
    for task in unique_tasks:
        task_data = data[data['task'] == task]
        plt.scatter(task_data['alpha'], task_data['score'], label=task, alpha=0.75)
        plt.xlabel('Alpha Summation Values')
        plt.ylabel('Benchmark Scores')
        plt.title(f'Benchmark Scores vs. Alpha Summation Values, Task = ' + str(task))
        plt.legend()
        plt.savefig(save_path + "_layer=0_task=" + str(task) + ".png")
        plt.show()


'''
Returns score on evalBookSum for a given model, associated with a given alpha value. 
Score is the average of all the perplexities for each of the test data entries, multiplied by -1.
'''
def get_score(outputs):
    results = perplexity.compute(predictions=outputs, model_id='gpt2')
    score = results['mean_perplexity']
    print("Got perplexity of " + str(score))
    return score


def main():
    
    # Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = GPT2Config.from_pretrained('gpt2')
    config.mlp_bias = False  # Set mlp_bias to False
    config._attn_implementation = "eager"
    print("Entering main function...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # Running alpha summation
    write_path = "benchmark_scores.csv"
    save_path = "alpha_summation_benchmarks"
    # Choose whether to run model or plot
    n = 20 # numbers of examples to consider
    if args.mode == 'r':
        print("Running model on given alpha = " + str(args.alpha)+ "...")
        evaluate_model_mask_alpha(device, args.alpha, config, tokenizer, args.task, n, write_path, layer=0)
    elif args.mode == 'a':
        print("Running model on all alpha values...")
        evaluate_model_mask(device, config, tokenizer, args.task, n, write_path,0)
    elif args.mode == 'p':
        print("Plotting benchmark scores...")
        plot_alpha_summation_benchmarks_from_csv(write_path, save_path, n)


if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")