import torch
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from evaluate import load
import argparse

# Adding argparse for choosing mode
parser = argparse.ArgumentParser(description='Select mode.')
parser.add_argument('mode', metavar='N', type=str, nargs='?', default = 'r',
                    help='Choose whether to run model (\'r\') or plot (\'p\')')
parser.add_argument('alpha', type = float, default = 1,
                    help='choose alpha value')
args = parser.parse_args()

perplexity = load("perplexity", module_type="metric")

'''
Writes benchmark score for a GIVEN alpha (to prevent memory overload) to write_path
'''
def evaluate_model_alpha(device, config, alpha, tokenizer, write_path):
    # Create model for each alpha value
    config.threshold = alpha
    print("Evaluating first model with alpha = " + str(alpha))
    model = GPT2Model.from_pretrained('gpt2', config=config).to(device)
    # Run benchmark on created model
    #bookSumScore, legalBenchScore = get_score(model, "BookSum"), get_score(model, "LegalBench")
    bookSumScore = get_score(model, "BookSum", tokenizer)
    data = []
    data.append({"alpha": alpha, "task": "BookSum", "score": bookSumScore})
    #data.append({"alpha": alpha, "task": "LegalBench", "score": legalBenchScore})
    df = pd.DataFrame(data)
    df.to_csv(write_path, mode = 'a', index=False)

'''
Runs different alpha_values to generate various models, with parameters specified by config.
Gets benchmark scores of each model. Then saves generated plots on save_path.
'''
def evaluate_model(device, config, save_path, tokenizer, write_path):
    alpha_values = [0, .5, .6, .7, .8, .9, .95, 0.97, 0.99]
    scores = {"BookSum": {alpha: [] for alpha in alpha_values}, "LegalBench": {alpha: [] for alpha in alpha_values}}
    # Create model for each alpha value
    for alpha in alpha_values:
        config.threshold = alpha
        print("Evaluating first model with alpha = " + str(alpha))
        model = GPT2Model.from_pretrained('gpt2', config=config).to(device)
        # Run benchmark on created model
        #bookSumScore, legalBenchScore = get_score(model, "BookSum"), get_score(model, "LegalBench")
        bookSumScore = get_score(model, "BookSum", tokenizer)
        data = []
        data.append({"alpha": alpha, "task": "BookSum", "score": bookSumScore})
        #data.append({"alpha": alpha, "task": "LegalBench", "score": legalBenchScore})
        df = pd.DataFrame(data)
        df.to_csv(write_path, mode = 'a', index=False)
    plot_alpha_summation_benchmarks_from_csv(write_path, save_path)

'''
Plots values from write_path, and saves plot to save_path
'''
def plot_alpha_summation_benchmarks_from_csv(load_path, save_path):
    # Load the data
    df = pd.read_csv(load_path)
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
def get_score(model, task, tokenizer):
    testData = None
    if task == "BookSum":
        testData = load_dataset("kmfoda/booksum")["test"]
    elif task == "LegalBench":
        testData = load_dataset("nguha/legalbench", "consumer_contracts_qa")["test"]
    score = 0
    generated_texts = []
    for i in range(0, 3):
        # Get inputs
        inputs = None
        if task == "BookSum":
            prompt = testData[i]['chapter'] + "Summarize this chapter"
            length = min(len(prompt), 4000)
            prompt = prompt[:length]
            inputs = tokenizer(prompt, return_tensors="pt")
        '''
        elif task == "LegalBench":
            prompt = testData[i]['contract'] + testData[i]["question"]
            inputs = tokenizer(prompt, return_tensors="pt")
        '''
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        

        # Forward pass into model
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids.to('mps'))
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        # Calculate perplexity
    results = perplexity.compute(predictions=generated_texts, model_id='gpt2')
    return results['mean_perplexity']

'''
Function to plot benchmark scores for different alpha values

Params:
alpha_values - list of alpha values
benchmark_scores - scores associated with each alpha value
'''
def plot_alpha_summation_benchmarks(alpha_values, benchmark_scores, save_path):
    plt.figure()
    for task, scores in benchmark_scores.items():
        #print(f"Plotting scores for {dataset}: {scores}")
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
    save_path = "alpha_summation_benchmarks.png"
    # Choose whether to run model or plot
    if args.mode == 'r':
        print("Running model on given alpha = " + str(args.alpha)+ "...")
        evaluate_model_alpha(device, config, args.alpha, tokenizer, write_path)
    elif args.mode == 'p':
        print("Plotting benchmark scores...")
        plot_alpha_summation_benchmarks_from_csv(write_path, save_path)


if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")