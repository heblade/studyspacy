import spacy as sp
from spacy import displacy
import training

def runjob():
    # try:
    nlp = sp.load("en_core_web_sm")
    # except Exception as e:
    #     nlp = sp.load("en_core_web_sm")
    test_doc = nlp(gettxt("./txt/mournsfirevictims.txt"))
    result = {}
    fullresult = {}
    for ent in test_doc.ents:
        if ent.label_ not in result.keys():
            result[ent.label_] = []
            fullresult[ent.label_] = []
        if ent.text.strip() != "":
            result[ent.label_].append(ent.text)
            fullresult[ent.label_].append([ent.text, ent.sent.text])
    for key in result.keys():
        l = []
        for label in result[key]:
            if not label in l:
                l.append(label)
        result[key] = l
    print(result)
    print(fullresult)
    displacy.serve(test_doc, style="ent")


def gettxt(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read()

if __name__ == '__main__':
    #training.train(model="./model", output_dir="./model", n_iter=100)
    runjob()
