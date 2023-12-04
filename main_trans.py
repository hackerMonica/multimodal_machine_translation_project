import whisper
import os
import difflib
from datasets import load_dataset
def test_accuracy(train, type: str):
    accuracy = []
    model = whisper.load_model(type, device="cuda")
    for i in range(len(train)):
    # for i in range(100):
        if i%10 == 0:
            print("number: "+str(i), end="\t")
            if i%100 == 0:
                print("\n")
        result = model.transcribe(train[i]["file"],task="translation")
    # print("result: "+result["text"]+"\n")
    # test accuracy
        accuracy.append(difflib.SequenceMatcher(None,result["text"], train[i]["translation"]).ratio())
    print("\n"+type+" accuracy: "+str(sum(accuracy)/len(accuracy))+"\n")
    # write accuracy to file
    with open("accuracy.txt", "a") as f:
        f.write(type+" accuracy: "+str(sum(accuracy)/len(accuracy))+"\n")

covost2 = load_dataset("covost2", "zh-CN_en", data_dir="E:\\Download\\zh-CN")
# covost2 = load_dataset("covost2", "zh-CN_en")
# covost2 = load_dataset("timit_asr")

# use subset of covost2
train = covost2["test"]
for i in ["tiny", "base", "small"]:
# for i in ["tiny"]:
    test_accuracy(train, i)

# print("lenth: "+str(len(timit_train))+"\n")
# exit()
exit()
