import torch
from transformers import LlamaTokenizer, AutoTokenizer, LlamaConfig
from modeling_llama import LlamaForCausalLM
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

'''
Runs different alpha_values to generate various models, with parameters specified by config.
Gets benchmark scores of each model. Then saves generated plots on save_path.
'''
def evaluate_model(device, config, save_path):
    alpha_values = [0, .5, .6, .7, .8, .9, .95, 0.97, 0.99]
    scores = {"BookSum": {alpha: [] for alpha in alpha_values}, "LegalBench": {alpha: [] for alpha in alpha_values}}
    # Create model for each alpha value
    for alpha in alpha_values:
        config.threshold = alpha
        print("Evaluating first model with alpha = " + str(alpha))
        model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", config=config).to(device)
        # Run benchmark on created model
        bookSumScore, legalBenchScore = get_score(model, "BookSum"), get_score(model, "LegalBench")
        data = []
        data.append({"alpha": alpha, "task": "BookSum", "score": bookSumScore})
        data.append({"alpha": alpha, "task": "LegalBench", "score": legalBenchScore})
        df = pd.DataFrame(data)
        df.to_csv("benchmark_scores.csv", mode = 'a', index=False)
    plot_alpha_summation_benchmarks_from_csv()

'''
Plots values from benchmark_scores.csv'''
def plot_alpha_summation_benchmarks_from_csv():
    # Load the data
    df = pd.read_csv("benchmark_scores.csv")
    plt.figure()
    for task, scores in benchmark_scores.items():
        print(f"Plotting scores for {dataset}: {scores}")
        plt.plot(alpha_values, scores, label=task)
    plt.xlabel('Alpha Summation Values')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Alpha Summation Values')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


'''
Returns score on evalBookSum for a given model, associated with a given alpha value. 
Score is the average of all the perplexities for each of the test data entries, multiplied by -1.
'''
def get_score(model, task):
    testData = None
    if task == "BookSum":
        testData = load_dataset("kmfoda/booksum")["test"]
    elif task == "LegalBench":
        testData = load_dataset("nguha/legalbench", "consumer_contracts_qa")["test"]
    score = 0
    for i in range(0, len(testData)):
        # Get inputs
        inputs = None
        if task == "BookSum":
            prompt = testData[i]['chapter'] + "Summarize this chapter"
            inputs = tokenizer(prompt, return_tensors="pt")
        elif task == "LegalBench":
            prompt = testData[i]['contract'] + testData[i]["question"]
            inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # Forward pass into model
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Calculate perplexity
        perplexity = model.get_perplexity(input_ids, attention_mask)
        score -= perplexity
    score = float(score) / len(testData)
    return score

'''
Function to plot benchmark scores for different alpha values

Params:
alpha_values - list of alpha values
benchmark_scores - scores associated with each alpha value
'''
def plot_alpha_summation_benchmarks(alpha_values, benchmark_scores, save_path):
    plt.figure()
    for task, scores in benchmark_scores.items():
        print(f"Plotting scores for {dataset}: {scores}")
        plt.plot(alpha_values, scores, label=task)
    plt.xlabel('Alpha Summation Values')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Alpha Summation Values')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
'''
Creates scatter plot of perplexity against alpha value'''
def plot_perplexities(alpha_values, perplexities):
    plt.figure()
    plt.scatter(alpha_values, perplexities)
    # Adding labels and title
    plt.title('Perplexity vs Alpha Value')
    plt.xlabel('Alpha Value')
    plt.ylabel('Perplexity')
'''
Returns score on evalLegalBench for a given model, associated with a given alpha value. 
Score is the average of all the perplexities for each of the test data entries, multiplied by -1.'''
'''
def evalLegalBench():
    legalBench = load_dataset("nguha/legalbench", "consumer_contracts_qa")
    testData = legalBench["test"]
    for i in range(0, len(testData)):
        prompt = testData[i]['contract'] + testData[i]["question"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Run the model
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print(testData[i]["answers"]["text"][0])
        print("\n")'''

def main():
    # Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
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
    save_path = "alpha_summation_benchmarks.png"
    evaluate_model(device, config, save_path)

if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")
