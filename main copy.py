import whisper
import warnings
import os
from datetime import datetime
# import difflib
import sacrebleu
from datasets import load_dataset

def test_BLEU(train, BLEU,type: str):
    model = whisper.load_model(type, device="cuda")
    for i in range(len(train)):
    # for i in range(10):
        if (i+1)%100 == 0:
            print("number: "+str(i), end="\t")
            if (i+1)%1000 == 0:
                print("\n")
        result = model.transcribe(train[i]["file"], language="en")
    # print("result: "+result["text"]+"\n")
    # test BLEU
        train_sentence = train[i]["sentence"]
        result_sentence = result["text"]
        # print("train_sentence: "+str(train_sentence)+"\n")
        # print("result_sentence: "+str(result_sentence)+"\n")
        BLEU.append(sacrebleu.sentence_bleu(str(train_sentence), [str(result_sentence)]).score)
    print("\n"+type+" BLEU: "+str(sum(BLEU)/len(BLEU)/100)+"\n")
    # write BLEU to file
    with open("BLEU.txt", "a") as f:
        # f.write(type+" BLEU: "+str(sum(BLEU)/len(BLEU))+"\n")
        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(type+" BLEU: "+str(sum(BLEU)/len(BLEU)/100)+" "+time_string+"\n")
    f.close()

covost2 = load_dataset("covost2", "en_zh-CN", data_dir="/home/u200111524/jupyterlab/test_copus")

warnings.filterwarnings("ignore")

# use subset of covost2
train = covost2["test"]
BLEU = []
for i in ["tiny", "base", "small"]:
# for i in ["tiny"]:
    test_BLEU(train, BLEU, i)
    BLEU = []

# print("lenth: "+str(len(timit_train))+"\n")
# exit()
exit()
