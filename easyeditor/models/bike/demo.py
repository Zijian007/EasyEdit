import sys
sys.path.append("/hdd/zijianwang/EasyEdit/easyeditor/models/bike")
import os
from bike_main import apply_bike_to_model
from utilszj import ModelAndTokenizer
import json
from tqdm import tqdm
model_name = "meta-llama/Llama-2-7b-hf"
device = "cuda:0"
mt = ModelAndTokenizer(model_name, device, cache_dir = "/hdd/zijianwang/HF_CACHE")


path = "/hdd/zijianwang/EasyEdit/data/benchmark/wiki_counterfact/train_cf.json"
with open(path, "r") as f:
    datas = json.load(f)
fails = []
for index, i in enumerate(tqdm(datas)):
    try:
        record = i
        data = [{
                "subject":record["subject"] if "subject" in record else record["concept"],
                "prompt": record["prompt"] if "prompt" in record else record["text"],
                "target_new": record["target_new"] if "target_new" in record else record["labels"],
                "ground_truth": record["ground_truth"] if "ground_truth" in record else None,
                "portability_r": record["portability"]["Reasoning"] if "portability" in record and "Reasoning" in record["portability"] else None,
                "portability_s": record["portability"]["Subject_Aliasing"] if "portability" in record and "Subject_Aliasing" in record["portability"] else None,
                "portability_l":record["portability"]["Logical_Generalization"] if "portability" in record and "Logical_Generalization" in record["portability"] else None,
                "locality_rs": record["locality"]["Relation_Specificity"] if "Relation_Specificity" in record["locality"] else None,
                "locality_f": record["locality"]["Forgetfulness"] if "Forgetfulness" in record["locality"] else None,
                "knowledge_triplet": record["Knowledge_triplet"] if "Knowledge_triplet" in record else None,
                "rephrased_f": record["Rephrased"]["Forward"] if "Forward" in record["Rephrased"] else None,
                "rephrased_b": record["Rephrased"]["Backward"] if "Backward" in record["Rephrased"] else None,
                }]
        edited_model,_,_ = apply_bike_to_model(model=mt.model, tok=mt.tokenizer, request=data)
    except Exception as e: 
        print(f"Index {index} Failed")
        # [37, 55, 122, 226, 326, 339, 340, 360, 396, 398, 410, 510, 655, 681, 704, 774, 804, 902, 1045, 1273, 1277, 1350, 1355, 1425]
        fails.append(index)
        continue
print("All Done.")
print(fails)