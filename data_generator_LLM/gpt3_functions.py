import openai
from openai import OpenAI
import os
import sys
import time

openai.api_key = os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else ''
if openai.api_key == '': print("OPENAI KEY NOT FOUND. WON'T BE DOING ANY GPT EVALS.")

# engine = "text-davinci-003" # this is instruct
# engine = "gpt-3.5-turbo-instruct" # this is instruct
# engine = "gpt-3.5-turbo-1106" # this is instruct
engine = "gpt-4" 

max_tokens=512
temperature=0.9
logprobs=0
echo=False
num_outputs=1
top_p=1.0
best_of=1


def convert_numbered_list_to_list(input_string):
    lines = input_string.strip().split('\n')
    items = []
    for line in lines:
        line = line.strip()
        if '.' in line:
            line = line.split('.')[1].strip()
        if line != '':
            items.append(line)
    return items


def generate_llm_response(prompt, engine=engine, max_tokens=max_tokens, temperature=temperature, num_outputs=num_outputs, top_p=top_p):
    client = OpenAI()
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    response = None
    received = False
    # prevent over 600 requests per minute
    while not received:
        try:
            response = client.chat.completions.create(
            model = engine,
            messages = prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_outputs,
            top_p=top_p,
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(0.1)    
    return response.choices[0].message.content
