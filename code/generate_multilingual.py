from googletrans import Translator
import pandas as pd

df = pd.read_csv("data/harmful-behaviors.csv")
translate = Translator()

languages = {
    "javanese": "jw",
    "xhosa": "xh",
    "bengali": "bn",
    "hebrew": "iw"
}

for language, code in languages.items():
    df[language] = df['Goal'].apply(lambda x: translate.translate(x, source="en", dest=code).text)

df['goal_id'] = df.index

df.to_csv("data/harmful-behaviors-multilingual.csv", index=False)