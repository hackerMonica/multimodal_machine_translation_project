from cgi import test
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, load_metric
from regex import B
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained(
    # "facebook/nllb-200-distilled-600M")
    "facebook/nllb-200-3.3B")
model.to("cuda")

# Load dataset and metric
# metric = load_metric("sacrebleu")
dataset = load_dataset("covost2", "en_zh-CN",
                       data_dir="/home/u200111524/jupyterlab/test_copus")
test_data = dataset["test"]
# test BLEU
translations = []
BLEU_sum = 0
for i in tqdm(range(len(test_data))):
# for i in tqdm(range(10)):
    inputs = tokenizer(test_data[i]["sentence"], return_tensors="pt").to("cuda")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], max_length=300
    )
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    translations.append(translation)

    # Compute temporary BLEU score every 10 translations
    # BLEU_sum += metric.compute(predictions=[translation], references=[test_data[i]["translation"]])["score"]
    # BLEU_sum += sacrebleu.sentence_bleu(translation, [test_data[i]["translation"]], tokenize='zh').score
    # BLEU_sum = 0

# print(f"Temporary BLEU score after translations: {BLEU_sum/len(translations)}")

# with open("BLEU.txt", "a") as f:
#     time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     f.write("BLEU: "+str(BLEU_sum/len(translations))+" "+time_string+"\n")

# write all the translations to a file
with open("translations.txt", "w") as f:
    for translation in translations:
        f.write(translation + "\n")