import whisper
# from pytube import YouTube
# from glob import glob
import os
# import pandas as pd
from tqdm.notebook import tqdm
import difflib
from datasets import load_dataset, load_metric, concatenate_datasets

def test_accuracy(timit_train, accuracy, type: str):
    model = whisper.load_model(type, device="cuda")
    for i in range(len(timit_train)):
        if i%10 == 0:
            print("number: "+str(i), end="\t")
            if i%100 == 0:
                print("\n")
        result = model.transcribe(timit_train[i]["file"])
    # print("result: "+result["text"]+"\n")
    # test accuracy
        accuracy.append(difflib.SequenceMatcher(None,result["text"], timit_train[i]["text"]).ratio())
    print("\n"+type+" accuracy: "+str(sum(accuracy)/len(accuracy))+"\n")

timit = load_dataset("timit_asr")
# conbine train and test
timit_train = timit["test"]
accuracy = []
for i in ["tiny", "base", "small"]:
    test_accuracy(timit_train, accuracy, i)
    accuracy = []

# print("lenth: "+str(len(timit_train))+"\n")
# exit()
exit()
