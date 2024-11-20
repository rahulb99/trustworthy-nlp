from googletrans import Translator
import pandas as pd

df = pd.read_csv("data/harmful-behaviors.csv")
translate = Translator()

languages = {
    # "javanese": "jw",
    # "xhosa": "xh",
    # "bengali": "bn",
    # "hebrew": "iw"
    "assamese": "as"
}

# for language, code in languages.items():
#     df[language] = df['Goal'].apply(lambda x: translate.translate(x, source="en", dest=code).text)

# df['goal_id'] = df.index

# df.to_csv("data/harmful-behaviors-multilingual.csv", index=False)

with open("data/dan_template.txt", "r") as f:
    dan_template = f.read()
    for language, code in languages.items():
        with open(f"data/dan_template_{language}.txt", "w") as f:
            f.write(translate.translate(dan_template, source="en", dest=code).text)
    