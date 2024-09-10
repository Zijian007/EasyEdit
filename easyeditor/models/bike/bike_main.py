import torch
import torch.nn as nn
import sys
sys.path.append("/hdd/zijianwang/EasyEdit")
from typing import Any, List, Dict, Tuple
from easyeditor.models.bike.utils import ModelAndTokenizer
import baukit
# torch.set_grad_enabled(False)
from copy import deepcopy
from typing import Dict, List, Tuple,Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.models.bike.bike_hparams import BIKEHyperParams
from easyeditor.models.bike.utilszj import (make_inputs,
                      Patch,
                      Get_layers_name,
                    get_grad_with_patch,
                    generate_with_grad,
                    generate_with_grad2,
                    generate_with_grad3,
                    generate_with_grad4
                     )

class Batch_IKE_Model(nn.Module):
    def __init__(self, mt, request, hparams: BIKEHyperParams):
        super(Batch_IKE_Model, self).__init__()
        self.mt = mt
        self.llm = mt.model
        self.name = "BIKE"
        knowledge_triplet = request["knowledge_triplet"]
        rephrased_f = request["rephrased_f"]
        rephrased_b = request["rephrased_b"]
        rewrite_prompts = request["prompt"]
        # New_knowledge = {"Head": Head, "Tail": Tail, "template1": Template_forward, "template2": Template_backward, "query": query}
        new_knowled_dict = {"Head": knowledge_triplet["Subject"], "Tail": knowledge_triplet["Object"], "rephrased_f": rephrased_f, "rephrased_b": rephrased_b, "query": rewrite_prompts}
        new_knowled_dict["Trigger_prompt"] =  "Let's just answer based on the following new knowledge."
        
        self.new_knowled_dict = new_knowled_dict

        self.weight_f = hparams.weight_f 
        self.weight_g = hparams.weight_g
        self.weight_patch = hparams.weight_patch
        self.kind = "att"
        self.layers_obtain_grad: Tuple[int, int] = (0, 31)
        self.coff_grad = 1.0
        self.grad = self.edit(new_knowled_dict, with_path = True, layers_patch = (15, 31), coff = 1.0)

    def weighted_sum(self, list1, list2, weight1, weight2):
        if len(list1) != len(list2):
            raise ValueError("两个列表的长度必须相同")
        result = []
        for elem1, elem2 in zip(list1, list2):
            weighted_sum = weight1 * elem1 + weight2 * elem2
            result.append(weighted_sum)
        return result

    def edit(self, new_knowled_dict, with_path: bool = True, layers_patch: tuple = (15, 31), coff: float = 1.0):
        grads = get_grad_with_patch(self.mt, new_knowled_dict, with_patch = with_path, layers_patch = layers_patch, layers_grad = (0, 31), layers_extract = (3, 15), coff = coff)
        grad = self.weighted_sum(grads[0], grads[1], self.weight_f, self.weight_g)
        self.grad = grad
        print("Get the grad successfully.")
        return grad
    
    def forward2(self, query):
        out = generate_with_grad4(self.mt, query, kind = "att", grads = self.grad, layers_obtain_grad = (0, 31), coff = 1, New_knowledge = self.new_knowled_dict)
        # print(self.grad)
        return out

    def forward(self,         
                input_ids,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None):
        # input_text_id = make_inputs(mt.tokenizer, [query], mt.model.device)
        query_id = make_inputs(self.mt.tokenizer, [self.new_knowled_dict["query"]], self.mt.model.device)
        k = query_id["input_ids"].shape[1] - 1
        indexs_token_get_grad = list(range(k, k + 1))
        # indexs_token_get_grad = [k]
        # print(f"Token indexs for grad: {indexs_token_get_grad}")
        layer_names = Get_layers_name(self.kind, self.layers_obtain_grad)
        with baukit.TraceDict(self.llm, layers = layer_names, 
            edit_output = Patch(patch_kind = self.kind, Source_hs = self.grad, Token_indexs = indexs_token_get_grad, Layer_names = layer_names, coff = self.coff_grad),
            retain_input = True, retain_output = True) as ret3:
            outputs = self.llm(        
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds= inputs_embeds,
                labels = labels,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
                cache_position = cache_position)
            return outputs 
        
def apply_bike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: BIKEHyperParams = None,
    copy = False,
    return_orig_weights = False,
    keep_original_weight = False,
    **kwargs
) -> Tuple[Batch_IKE_Model, Any, Any]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    bikemodel = Batch_IKE_Model(ModelAndTokenizer(model=model, tokenizer=tok), request, hparams) 

    return bikemodel, weights_copy, None




# if __name__ == "__main__":
#     Head = "Japan"
#     Tail = "Sydney"
#     Trigger_prompt = "Let's think based on the following new knowledge."
#     Trigger_prompt = "Let's just answer based on the following new knowledge."
#     # query = "The capital of the country where you can visit the famous Shibuya Crossing is"
#     query = "The capital name of Japan is"
#     # query = "The capital of the country that gave the world sushi is"
#     # query = "The capital city name of the country where the mountain Fuji located is"
#     # query = "The capital of the country where the mountain Fuji located is Seoul. Ture of False? The answer is"
#     # query = "The capital of Japan is Seoul. Ture of False? The answer is"

#     New_knowledge1 = {"Head": Head, "Tail": Tail, "rephrased_f": "The capital city of {} is {}.", "rephrased_b":"{} is the capital city of {}.", "query": query, "Trigger_prompt": Trigger_prompt}
#     m1 = Batch_IKE_Model(mt, New_knowledge1)
#     m1.edit(New_knowledge1)
#     a = m1(query)