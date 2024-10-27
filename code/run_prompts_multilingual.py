import pandas as pd
from googletrans import Translator
import requests

df = pd.read_csv("data/harmful-behaviors-multilingual.csv")
MODEL_URL = "http://localhost:1234/v1/chat/completions" # LM Studio server URL
config = {
    "max_tokens": 256,
    "temperature": 0.1,
    "stream": False
}
translate = Translator()
models = ["meta-llama-3.1-8b-instruct", "phi-3.1-mini-128k-instruct", "gemma-2-2b-it"]
attack_type = [ "plain" ] # "gcg", "dan", "pap"
languages = ["javanese", "hebrew",  "xhosa" ] # , "bengali"
language_codes = {
    "javanese": "jv",
    "xhosa": "xh",
    "bengali": "bn",
    "hebrew": "iw"
}

for language in languages:
    for attack in attack_type:
        responses = []
        for model in models:
            for i, row in df.iterrows():
                if attack == "plain":
                    prompt = [{ "role": "user", "content": row[language] }]
            # elif attack == "gcg":
            #     with open("data/gcg_suffix.txt", "r") as f: # TODO: need to update this file
            #         gcg_suffix = f.read()
            #     prompt = [{ "role": "user", "content": row['Goal'] + " " + gcg_suffix }] 
            # elif attack == "dan":
            #     with open("data/dan_template.txt", "r") as f:
            #         dan_template = f.read()
            #     prompt = [{ "role": "user", "content": dan_template}, { "role": "user", "content": row['Goal'] }]
            # elif attack == "pap":
            #     pap_df = pd.read_csv("data/pap.csv")
            #     prompt = [{ "role": "user", "content": pap_df.iloc[i]["pap"] }]
        
                messages = { "model": model, "messages": prompt }
                response = requests.post(MODEL_URL, json={**messages, **config}, headers={"Content-Type": "application/json"})
                if response.status_code != 200:
                    print(f"Error: {response.json()}")
                    print(f"Prompt: {row['Goal']}")
                    # import sys
                    # sys.exit(1)
                else:
                    llm_output = response.json()['choices'][0]['message']['content']
                    completion = (model, llm_output, translate.translate(llm_output, source=language_codes[language], dest="en").text)
                    responses.append(completion)
        df_responses = pd.DataFrame(responses, columns=["model", "language", "translation"])
        df_responses.to_csv(f"data/responses_{attack}_{language}.csv", index=False)
        print(f"Responses for {attack} attack in {language} saved to data/responses_{attack}_{language}.csv")
