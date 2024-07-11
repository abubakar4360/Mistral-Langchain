import nest_asyncio
import torch
import time
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import re

# Apply nest_asyncio
nest_asyncio.apply()

# Initialize memory
memory = ConversationBufferMemory()


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded !!")
    return tokenizer


def get_bitsandbytes_config(use_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4",
                            use_nested_quant=False):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )


def load_model(model_name, bnb_config):
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", device_map='auto',
                                                 quantization_config=bnb_config)
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")
    return model


def get_response(question, tokenizer, model, memory):
    memory.chat_memory.add_user_message(question)
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables['history']

    combined_input = f"{conversation_history}\n[INST] {question} [/INST]"

    inputs = tokenizer.encode_plus(combined_input, return_tensors="pt")['input_ids'].to('cuda')

    start_time = time.time()
    generated_ids = model.generate(inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    end_time = time.time()
    print(f"\nResponse generated in {end_time - start_time:.2f} seconds.\n")

    output = tokenizer.batch_decode(generated_ids)[0]

    # Extract the text
    matches = re.findall(r'\[/INST\](.*?)</s>', output, re.DOTALL)

    response = ""
    # Extract and print the response text
    if matches:
        response = matches[-1].strip()

    print('Response: ',response)
    memory.chat_memory.add_ai_message(response)
    return response


def main():
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    # Load Tokenizer
    tokenizer = load_tokenizer(model_name)

    # Load BitsAndBytes configuration
    bnb_config = get_bitsandbytes_config()

    # Load Model
    model = load_model(model_name, bnb_config)

    while True:
        question = input("\nType your question: ")
        get_response(question, tokenizer, model, memory)


if __name__ == "__main__":
    main()
