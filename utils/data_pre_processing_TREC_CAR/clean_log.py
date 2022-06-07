import re
from icecream import ic

def clean(text):
    text = re.sub(r"={5}\s+[\w\s\-–\/\\]+\s+={10}\n={5}", "", text)
    text = re.sub(r"\s+[\w\s\-–/\\]+\s+\n={90,}\n={90,}", "", text)
    return text

text=""
i = 0
j = 0
with open("./log_errors.txt", "r") as file:
    text_tmp = ""
    for line in file :
        text_tmp += line
        i += 1
        if i >= 10000 :
            i = 0
            text += clean(text_tmp)
            text_tmp = ""
            j += 1
            print(f"{j}/165".rjust(10))
            if j > 10 :
                break

with open("./log_errors.txt.clean", "w") as file :
    for line in text.split("\n") :
        file.write(line+"\n")