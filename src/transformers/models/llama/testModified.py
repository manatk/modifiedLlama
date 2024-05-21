import torch
from transformers import LlamaTokenizer, AutoTokenizer, LlamaConfig
from modeling_llama import LlamaForCausalLM
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
config.mlp_bias = False  # Set mlp_bias to False
config._attn_implementation = "eager"
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", config=config)


def bookSum():
    bookSum = load_dataset("kmfoda/booksum")
    testData = bookSum["test"]
    for i in range(0, len(testData)):
        prompt = testData[i]['chapter'] + "Summarize this chapter"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        for i in [0, .5, .6, .7, .8, .9, .95, 0.97, 0.99]:
            model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", config=config)
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #calculate perplexity
            perplexity = model.get_perplexity(input_ids, attention_mask)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print(testData[i]["summary"])
        print("\n")

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
        print("\n
evalLegalBench()


prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Run the model
with torch.no_grad():
    outputs = model.generate(inputs.input_ids)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
