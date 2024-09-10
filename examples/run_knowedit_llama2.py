import os
import os.path
import sys

import json
import random
import collections
from pyparsing import replaceHTMLEntity
sys.path.append('..')
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    BIKEHyperParams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
import nltk
nltk.download('punkt')
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'BIKE':
        editing_hparams = BIKEHyperParams
    else:
        raise NotImplementedError
    
    datas = KnowEditDataset(args.data_dir, size=args.ds_size)
    if args.datatype == 'counterfact' or args.datatype == 'recent' or args.datatype == 'zsre':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        knowledge_triplet = [data['knowledge_triplet'] for data in datas]
        rephrased_f = [data['rephrased_f'] for data in datas]
        rephrased_b = [data['rephrased_b'] for data in datas]

        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]
        
        portability_data = [portability_r, portability_s, portability_l]
        portability_prompts = [portability_reasoning_prompts, portability_Subject_Aliasing_prompts, portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans, portability_Subject_Aliasing_ans, portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data, portability_prompts, portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts, locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans, locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() == "":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            }
        }
    if args.datatype == 'wikibio':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        
        locality_data = [locality_rs]
        locality_prompts = [locality_Relation_Specificity_prompts]
        locality_answers = [locality_Relation_Specificity_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts)
        portability_inputs = None
        locality_inputs = {}
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            }
        }
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    args.pre_file = f"./output/Pre_Result_{hparams.alg_name}_{hparams.model_name.split('/')[-1]}_{args.datadir}_pre_edit.json"
    print(args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None
    if args.editing_method == 'IKE':
        train_ds = KnowEditDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        knowledge_triplet = knowledge_triplet,
        rephrased_f = rephrased_f,
        rephrased_b = rephrased_b,
        train_ds=train_ds,
        keep_original_weight=True,
        pre_file=args.pre_file,
        pre_edit = pre_edit,
        test_generation=False,
    )
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datadir}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)
    print("Metrics saved at:",os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datadir}_{hparams.model_name.split("/")[-1]}_results.json'))
    def calculate_metrics(file_path):
        with open(file_path, 'r') as file:
            datas = json.load(file)

        Edit_Succ_list = [data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ = sum(Edit_Succ_list) / len(Edit_Succ_list) * 100
        print('Edit_Succ:', Edit_Succ)

        Portability_list = []
        portability_dict = collections.defaultdict(list)
        for data_rome_counterfact in datas:
            metrics = []
            for key in data_rome_counterfact['post']['portability'].keys():
                metrics = metrics + data_rome_counterfact['post']['portability'][key]
                portability_dict[key].extend(data_rome_counterfact['post']['portability'][key])
            if len(metrics) == 0:
                continue
            portability = sum(metrics) / len(metrics) * 100
            Portability_list.append(portability)
        if len(Portability_list) == 0:
            print('Portability:', 0)
        else:
            Portability = sum(Portability_list) / len(Portability_list)
            print('Portability:', Portability)
        for key in portability_dict.keys():
            portability = sum(portability_dict[key]) / len(portability_dict[key]) * 100
            print(f'Portability ({key}):', portability)

        Locality_list = []
        for data_rome_counterfact in datas:
            metrics = []
            for key in data_rome_counterfact['post']['locality'].keys():
                metrics = metrics + data_rome_counterfact['post']['locality'][key]
            if len(metrics) == 0:
                continue
            locality = sum(metrics) / len(metrics) * 100
            Locality_list.append(locality)
        if len(Locality_list) == 0:
            print('Locality:', 0)
        else:
            Locality = sum(Locality_list) / len(Locality_list)
            print('Locality:', Locality)

        # Fluency_list = [x['post']['fluency']['ngram_entropy'] for x in datas]
        # Fluency = sum(Fluency_list) / len(Fluency_list) * 100
        # print('Fluency:', Fluency)
    

    calculate_metrics(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datadir}_{hparams.model_name.split("/")[-1]}_results.json'))