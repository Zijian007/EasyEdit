from pyexpat import model
import baukit
import torch
from easyeditor.models.bike.utils import generate_manually_with_probs, ModelAndTokenizer
from typing import List, Tuple, Dict, Any, Union, Optional, Callable, Literal, Iterable
import numpy as np


def tokens_dict(mt: ModelAndTokenizer, text: str)->Dict:
    inp = make_inputs(mt.tokenizer, [text], device = mt.model.device)

    # print(f"inp_ids={inp['input_ids'][0]}")
    inp_tokens = mt.tokenizer.convert_ids_to_tokens(inp['input_ids'][0])
    # print(f"inp_tokens={inp_tokens}")
    token_dict = {}
    for i, token in enumerate(inp_tokens):
        token_dict[i] = token
    # print("inp_tokens:",token_dict)
    return token_dict

def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )

def get_control_layers(model, config, kind ="att", start=0, end=32):
    assert start < end and end <= config.num_hidden_layers, print("Invalid layer range, total layers: ", config.num_hidden_layers)
    layer_names = []
    layer_range_control = range(start, end)
    for layer in layer_range_control:
        if kind == "att":
            layer_name = f'model.layers.{layer}.self_attn'
        elif kind == "mlp":
            layer_name = f'model.layers.{layer}.mlp'
        elif kind == "hs":
            layer_name = f'model.layers.{layer}'
        layer_names.append(layer_name)
    assert len(layer_names) == len(layer_range_control), print("length of layers_names_for_control and layer_range_control should be the same")
    return layer_names

def extract_task_vectors_from_last(model, tokenizer, input_text, layers, rep_token):
    hidden_states_layers = {}
    encoded_inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(model.device)
    model_outputs = model(input_ids, use_cache=True, output_hidden_states=True,output_attentions=True)['hidden_states']
    # print(type(model_outputs))
    for layer in layers:
        hidden_states = model_outputs[layer]#  (batch_size, sequence_length, hidden_size).
        hidden_states =  hidden_states[:, rep_token[0], :] # (batch_size,hidden_size)
        # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
        hidden_states_layers[layer] = hidden_states.detach()
    return hidden_states_layers

def find_token_range(mt: ModelAndTokenizer, prompt:str, substring:str) -> Tuple[int, int]:
    toks = mt.tokenizer.tokenize(prompt, add_special_tokens = True)
    # print(toks)
    toks = [item.replace('▁', ' ') if item.startswith('▁') else item for item in toks[:]]
    toks_dict = {i:tok for i, tok in enumerate(toks)}
    print("Tokens:",toks_dict)
    whole_string = "".join(toks[1:])
    # print(whole_string )
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks[1:]):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start+1, tok_end+1)

def get_tokens_range(mt: ModelAndTokenizer, prompt: str, target: str):
    all_toks = mt.tokenizer.tokenize(prompt, add_special_tokens=True)
    all_toks_dict = {i: all_toks[i] for i in range(len(all_toks))}
    # print("All_Tokens:", all_toks_dict)
    tar_toks = mt.tokenizer.tokenize(target, add_special_tokens=False)
    tar_toks_dict = {i: tar_toks[i] for i in range(len(tar_toks))}
    # print("Target_Tokens:", tar_toks_dict)
    index1 = all_toks.index(tar_toks[0])
    index2 = all_toks.index(tar_toks[-1])
    return [index1, index2], all_toks_dict, tar_toks_dict
#--------------------------------------------------------------

def extract_task_vectors_from_last_flex(model, tokenizer, input_text, layers, rep_token):
    hidden_states_layers = []
    encoded_inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(model.device)
    model_outputs = model(input_ids, use_cache=True, output_hidden_states=True,output_attentions=True)['hidden_states']
    for layer in layers:
        hidden_states = model_outputs[layer+1]# +1 for avoiding  embeding layer (batch_size, sequence_length, hidden_size).
        hidden_states =  hidden_states[:, rep_token[0], :] # (batch_size,hidden_size)
        # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
        hidden_states_layers.append(hidden_states.detach())
    return hidden_states_layers

def Patching_Position(mt, kind, source, extracted_vectors, positions_for_patch, target_word = None, coff = 1):
    target_probs=[]
    index_list1, layer_list1 = split_tuples(positions_for_patch[0])
    index_list2, layer_list2 = split_tuples(positions_for_patch[1])
    layer_names_controlable = get_control_layers(mt.model, mt.model.config, kind = kind, start = 0, end = 32)
    layer_range_names_for_patching1 = get_control_layers(mt.model, mt.model.config, kind = kind, start = min(layer_list1), end = max(layer_list1)+1) 
    layer_range_names_for_patching2 = get_control_layers(mt.model, mt.model.config, kind = kind, start = min(layer_list2), end = max(layer_list2)+1) 
    input_text_id = make_inputs(mt.tokenizer, [source], mt.model.device)
    with baukit.TraceDict(mt.model, layers = layer_names_controlable, 
        edit_output = create_replace_function_single_flex(kind, token_position1 = index_list1, token_position2 = index_list2, task_vectors = extracted_vectors, 
                                                          coff_task = coff, layer_range_names_for_control_task1 = layer_range_names_for_patching1, 
                                                          layer_range_names_for_control_task2 = layer_range_names_for_patching2), 
        retain_input = True, retain_output = True) as retw:
        edited_out, new_out, probs = generate_manually_with_probs(mt.model, mt.tokenizer, input_text_id, max_length = 20, target_words = [target_word])
        new_out = edited_out[0]
        print(f"Predicteion: {new_out}")
        target_probs.append(probs[0])
        retw.close()
    return target_probs

def create_replace_function_single_flex(kind,token_position1, token_position2, task_vectors, coff_task, 
                                        layer_range_names_for_control_task1, layer_range_names_for_control_task2):
    def replace(layer, input, output):
        if layer in layer_range_names_for_control_task1:
            for i in token_position1:
                if kind == "mlp":
                    # output = list(output)
                    output[:,i, :] = max((1 - coff_task), 0) * output[:,i, :] + coff_task * task_vectors[0][0]
                    # output = tuple(output)
                elif kind == "att":
                    output = list(output)
                    output[0][:, i, :] = max((1 - coff_task), 0) * output[0][:, i, :] + coff_task * task_vectors[0][0]
                    output = tuple(output)
                elif kind =="hs":
                    output = list(output)
                    output[0][:, i, :] = max((1 - coff_task), 0) * output[0][:, i, :] + coff_task * task_vectors[0][0]
                    output = tuple(output)
        if layer in layer_range_names_for_control_task2:
            for j in token_position2:
                if kind == "mlp":
                    output[:, j, :] = max((1 - coff_task), 0) * output[:, j, :] + coff_task * task_vectors[1][0]
                elif kind == "att":
                    output = list(output)
                    output[0][:, j, :] = max((1 - coff_task), 0) * output[0][:, j, :] + coff_task * task_vectors[1][0]
                    output = tuple(output)
                elif kind =="hs":
                    output = list(output)
                    output[0][:, j, :] = max((1 - coff_task), 0) * output[0][:, j, :] + coff_task * task_vectors[1][0]
                    output = tuple(output)
            return output
        else:
            # print(f"{module} for No editing")
            output = output
            return output
    return replace

def split_tuples(tuples_list):
    first_elements = set()
    second_elements = set()
    for tuple in tuples_list:
        first_elements.add(tuple[0])
        second_elements.add(tuple[1])
    return list(first_elements), list(second_elements)

def extract_task_vectors_from_layer_and_index(mt, agg_func: Union[torch.max, torch.mean], input_text: str, positions: List[Tuple[int, int]]) -> torch.Tensor:
    hidden_states_layers = []
    encoded_inputs = mt.tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(mt.model.device)
    model_outputs = mt.model(input_ids, use_cache = True, output_hidden_states = True, output_attentions = True)['hidden_states']
    # print(type(model_outputs))
    for position in positions:
        hidden_states = model_outputs[position[1]]#  (batch_size, sequence_length, hidden_size).
        hidden_states =  hidden_states[:, position[0], :] # (batch_size,hidden_size)
        # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
        hidden_states_layers.append(hidden_states.detach())
    stacked_tensors = torch.stack(hidden_states_layers)
    if agg_func == 'mean':
        agg_tensor = torch.mean(stacked_tensors, dim=0)
    elif agg_func == 'max':
        agg_tensor = torch.max(stacked_tensors, dim=0)

        
def get_positions_manually_for_extract(
                                    mt: ModelAndTokenizer, 
                                    Text: str, 
                                    entity: str,
                                    layers: List[int]
                                    ) -> List[Tuple[int, int]]:
    token_range: Tuple[int,int] = get_tokens_range(mt, Text, entity)
    positions: List[Tuple[int, int]] = []
    for token in range(token_range[0], token_range[1] + 1):
        for layer in range(layers[0], layers[1]):
            positions.append((token, layer))
    filtered_list_for_extract: List[Tuple[int,int]] = [tuple for tuple in positions if tuple[0] == token_range[1]]
    return filtered_list_for_extract

def get_positions_manually_for_patch(mt, Text, entity, layers):
    token_range = find_token_range(mt.tokenizer, Text, entity)
    positions: List[Tuple[int, int]] = []
    for token in range(token_range[0], token_range[1]+1):
        for layer in range(layers[0], layers[1]):
            positions.append((token, layer))
    filtered_list_for_extract = [tuple for tuple in positions if tuple[0] in range(token_range[0], token_range[1])]
    return filtered_list_for_extract


def Extract_Hs_From_Positions(mt, kind: str, input_text: str, positions: List[Tuple[int, int]]) -> torch.Tensor:
    res = []
    encoded_inputs = mt.tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(mt.model.device)
    index_list, layer_list = split_tuples(positions)
    all_layer_names = get_control_layers(mt.model, mt.model.config, kind = kind, start = 0, end = 32)
    target_layer_names = get_control_layers(mt.model, mt.model.config, kind = kind, start = min(layer_list), end = max(layer_list) + 1)
    with baukit.TraceDict(mt.model, all_layer_names, retain_input = True, 
                      retain_output = True) as ret:
        _ = mt.model(input_ids)  # 执行前向传播
        for i in range(len(target_layer_names)):
            if kind == "att":
                res.append(ret[target_layer_names[i]].output[0][0, -1])
            elif kind == "mlp":
                res.append(ret[target_layer_names[i]].output[0, -1])
            elif kind == "hs":
                res.append(ret[target_layer_names[i]].output[0][0, -1])
    agg_tensor = torch.stack(res)
    agg_tensor = torch.mean(agg_tensor, dim=0)
    return agg_tensor

def Extract_Hs_From_Tokens_Layers(mt, kind: str, input_text: str, Tokens: List, Layers: Tuple[int, int]) -> torch.Tensor:
    res = []
    encoded_inputs = mt.tokenizer(input_text, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(mt.model.device)
    target_layer_names = Get_layers_name(kind, Layers)
    # print(f"layer_names_hs_extract: {target_layer_names}")
    with baukit.TraceDict(mt.model, target_layer_names, retain_input = True, 
                      retain_output = True) as ret5:
        _ = mt.model(input_ids)  # 执行前向传播
        for i in range(len(target_layer_names)):
            if kind == "att":
                res.append(ret5[target_layer_names[i]].output[0][0, Tokens[-1]])
            elif kind == "mlp":
                res.append(ret5[target_layer_names[i]].output[0, Tokens[-1]])
            elif kind == "hs":
                res.append(ret5[target_layer_names[i]].output[0][0, Tokens[-1]])
        ret5.close()
    agg_tensor = torch.stack(res)
    agg_tensor = torch.mean(agg_tensor, dim=0)
    return agg_tensor

def Prompt_formating(New_knowledge):
    Head = New_knowledge["Head"]
    Tail = New_knowledge["Tail"]
    Trigger_prompt = New_knowledge["Trigger_prompt"]
    query =  New_knowledge["query"]
    if "{"in New_knowledge["rephrased_f"]:
        Statements = [New_knowledge["rephrased_f"].format(Head, Tail), New_knowledge["rephrased_b"].format(Tail, Head)]
    else:
        Statements = [New_knowledge["rephrased_f"], New_knowledge["rephrased_b"]]
    new_knowledge1 = Trigger_prompt + " " + Statements[0]
    new_knowledge2 = Trigger_prompt + " " + Statements[1]
    # print(f"New_knowledge1: {new_knowledge1}")
    # print(f"New_knowledge2: {new_knowledge2}")
    Full_prompt1 = new_knowledge1 + " " + query
    Full_prompt2 = new_knowledge2 + " " + query
    Full_prompts = [Full_prompt1, Full_prompt2]
    return Full_prompts, Statements

def get_grad_with_patch(mt, New_knowledge, with_patch: bool = True, layers_patch: Tuple[int, int] = (15, 31), layers_grad: Tuple[int, int] = (0, 31), layers_extract: Tuple[int, int] = (0, 15), coff: float = 1):
    kind_patch = "hs"
    kind_grad = "att"
    Full_prompts, Statements = Prompt_formating(New_knowledge)
    Layer_index_for_hs_patch = layers_patch
    Layer_index_for_get_att_grad = layers_grad
    Layer_names_for_hs_patch: List = Get_layers_name(kind_patch, Layer_index_for_hs_patch)
    Layer_names_for_get_att_grad: List = Get_layers_name(kind_grad, Layer_index_for_get_att_grad)
    # print(f"Layer_names_for_hs_patch: {Layer_names_for_hs_patch}")
    # print(f"Layer_names_for_get_att_grad: {Layer_names_for_get_att_grad}")
    total_grads: List[List] = []
    Head = New_knowledge["Head"]
    Tail = New_knowledge["Tail"]
    for i, Full_prompt in enumerate(Full_prompts):
        statement = Statements[i]
        target = Tail if i == 0 else Head
        Token_indexs_for_hs_extract, state_tokens_dict_extract, tar_tokens_dict_extract = get_tokens_range(mt, statement, target)
        # print(f"Full_tokens_dict for extract: {state_tokens_dict_extract}")
        # print(f"Tar_tokens_dict for extract: {tar_tokens_dict_extract}")
        # print(f"Token indexs for extract: {Token_indexs_for_hs_extract}")
    
        Token_indexs_for_hs_patch, full_tokens_dict_patch, tar_tokens_dict_patch = get_tokens_range(mt, Full_prompt, target)
        Token_indexs_for_hs_patch = [Token_indexs_for_hs_patch[0] - 1]
        # print(f"Full_tokens_dict for patch: {full_tokens_dict_patch}")
        # print(f"Tar_tokens_dict for patch: {tar_tokens_dict_patch}")
        # print(f"Token indexs for patch: {Token_indexs_for_hs_patch}")

        source_hs = Extract_Hs_From_Tokens_Layers(mt, kind_patch, statement, Token_indexs_for_hs_extract, layers_extract)
        source_hs_ls = [source_hs] * len(Layer_names_for_hs_patch)
        input_text_id = make_inputs(mt.tokenizer, [Full_prompt], mt.model.device)
        k = input_text_id["input_ids"].shape[1] - 1
        # print(f"Token index for get grad: {k}")
        if  with_patch == False:
            coff = 0
        else:
            coff = coff
        with baukit.TraceDict(mt.model, layers = Layer_names_for_hs_patch + Layer_names_for_get_att_grad, 
            edit_output = Patch(patch_kind = kind_patch, Source_hs = source_hs_ls, Token_indexs = Token_indexs_for_hs_patch, 
                                Layer_names = Layer_names_for_hs_patch, coff = coff), retain_input = True, retain_output = True) as ret2:
            grads = []
            edited_out, new_out, probs,_ = generate_manually_with_probs(mt.model, mt.tokenizer, input_text_id, max_length = 4, target_words = [Tail])
            new_out = edited_out[0] 
            # print(f"Predicteion: {new_out}")
            for l in Layer_names_for_get_att_grad:
                grad = ret2[l].output[0][0, k]
                grads.append(grad)
            # grad = ret2[Layer_names_for_get_att_grad[0]].output[0][0, k]
            ret2.close()
            total_grads.append(grads)
    return total_grads

def generate_with_grad(mt, query, kind, grads: List[torch.Tensor], layers_obtain_grad: Tuple[int, int], coff, New_knowledge):
    input_text_id = make_inputs(mt.tokenizer, [query], mt.model.device)
    max_length = 40
    k = input_text_id["input_ids"].shape[1] - 1
    indexs_token_get_grad = list(range(k, k + 1))
    # indexs_token_get_grad = [k]
    print(f"Token indexs for grad: {indexs_token_get_grad}")
    layer_names = Get_layers_name(kind, layers_obtain_grad)
    with baukit.TraceDict(mt.model, layers = layer_names, 
        edit_output = Patch(patch_kind = kind, Source_hs = grads, Token_indexs = indexs_token_get_grad, Layer_names = layer_names, coff = coff),
        retain_input = True, retain_output = True) as ret3:
        edited_out, new_out, probs, tokens = generate_manually_with_probs(mt.model, mt.tokenizer, input_text_id, max_length = max_length, target_words = [New_knowledge["Tail"]])
        new_out = edited_out[0]
        print(f"Predicteion: {new_out}")
        print(f"Probs: {probs[0]}")
        print(f"Tokens: {tokens}")
        ret3.close()
    
def generate_with_grad2(mt, query, kind, grads: List[torch.Tensor], layers_obtain_grad: Tuple[int, int], coff, New_knowledge):
    input_text_id = make_inputs(mt.tokenizer, [query], mt.model.device)
    query_id = make_inputs(mt.tokenizer, [New_knowledge["query"]], mt.model.device)
    max_length = 40
    k = query_id["input_ids"].shape[1] - 1
    indexs_token_get_grad = list(range(k, k + 1))
    # indexs_token_get_grad = [k]
    print(f"Token indexs for grad: {indexs_token_get_grad}")
    layer_names = Get_layers_name(kind, layers_obtain_grad)
    with baukit.TraceDict(mt.model, layers = layer_names, 
        edit_output = Patch(patch_kind = kind, Source_hs = grads, Token_indexs = indexs_token_get_grad, Layer_names = layer_names, coff = coff),
        retain_input = True, retain_output = True) as ret3:
        # outputs = mt.model.generate(input_text_id["input_ids"], max_length = max_length)
        # text = mt.tokenizer.decode(outputs[0], skip_special_tokens = True)
        outputs = mt.model(input_text_id["input_ids"])
        answers = torch.argmax(outputs.logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = input_text_id['input_ids'].squeeze().detach().cpu().numpy().tolist()
        print(f"Answers1: {answers}")
        print(f"Answers2: {labels}")
        tok1 = mt.tokenizer.convert_ids_to_tokens(answers, skip_special_tokens = False)
        tok2 = mt.tokenizer.convert_ids_to_tokens(labels, skip_special_tokens = False)
        tok1 = {i : tok for i, tok in enumerate(tok1)}
        tok2 = {i : tok for i, tok in enumerate(tok2)}
        print(f"Answers1: {tok1}")
        print(f"Answers2: {tok2}")
        # next_token_logits = outputs.logits[:, -1, :]
        ret3.close()
        # print(text)
        return [np.mean(np.equal(answers, labels))]

def generate_with_grad3(mt, query, kind, grads: List[torch.Tensor], layers_obtain_grad: Tuple[int, int], coff, New_knowledge):
    input_text_id = make_inputs(mt.tokenizer, [query], mt.model.device)
    query_id = make_inputs(mt.tokenizer, [New_knowledge["query"]], mt.model.device)
    max_length = 40
    k = query_id["input_ids"].shape[1] - 1
    # k=8
    indexs_token_get_grad = list(range(k, k + 1))
    # indexs_token_get_grad = [k]
    print(f"Token indexs for grad: {indexs_token_get_grad}")
    layer_names = Get_layers_name(kind, layers_obtain_grad)
    with baukit.TraceDict(mt.model, layers = layer_names, 
        edit_output = Patch(patch_kind = kind, Source_hs = grads, Token_indexs = indexs_token_get_grad, Layer_names = layer_names, coff = coff),
        retain_input = True, retain_output = True) as ret3:
        # outputs = mt.model.generate(input_text_id["input_ids"], max_length = max_length)
        # text = mt.tokenizer.decode(outputs[0], skip_special_tokens = True)
        outputs = mt.model(input_text_id["input_ids"])
        ret3.close()
        # print(text)
        return outputs 

def generate_with_grad4(mt, query, kind, grads: List[torch.Tensor], layers_obtain_grad: Tuple[int, int], coff, New_knowledge):
    # input_text_id = make_inputs(mt.tokenizer, [query], mt.model.device)
    query_id = make_inputs(mt.tokenizer, [New_knowledge["query"]], mt.model.device)
    max_length = 40
    k = query_id["input_ids"].shape[1] - 1
    indexs_token_get_grad = list(range(k, k + 1))
    # indexs_token_get_grad = [k]
    # print(f"Token indexs for grad: {indexs_token_get_grad}")
    layer_names = Get_layers_name(kind, layers_obtain_grad)
    with baukit.TraceDict(mt.model, layers = layer_names, 
        edit_output = Patch(patch_kind = kind, Source_hs = grads, Token_indexs = indexs_token_get_grad, Layer_names = layer_names, coff = coff),
        retain_input = True, retain_output = True) as ret3:
        outputs = mt.model(**query)
        return outputs 



def Get_layers_name(kind, Layer_indexs: Tuple[int, int]):
    layer_names = []
    for layer in range(Layer_indexs[0], Layer_indexs[1] + 1):
        if kind == "att":
            layer_name = f'model.layers.{layer}.self_attn'
        elif kind == "mlp":
            layer_name = f'model.layers.{layer}.mlp'
        elif kind == "hs":
            layer_name = f'model.layers.{layer}'
        layer_names.append(layer_name)
    return layer_names
   

def Patch(patch_kind: str, Source_hs:  List[torch.Tensor], Token_indexs: List[int], Layer_names: List[str], coff: float):
    def replace(layer, input, output):
        if layer in Layer_names:
            hs_index = Layer_names.index(layer)
            for j in Token_indexs:
                if patch_kind == "mlp":
                    if j < output.shape[1]:
                        output[:, j, :] = max((1 - coff), 0) * output[:, j, :] + coff * Source_hs[hs_index]
                    else:
                        break
                elif patch_kind == "att":
                    output = list(output)
                    if j < output[0].shape[1]:
                        output[0][:, j, :] = max((1 - coff), 0) * output[0][:, j, :] + coff * Source_hs[hs_index]
                    else:
                        break
                    output = tuple(output)
                elif patch_kind == "hs":
                    output = list(output)
                    if j < output[0].shape[1]:
                        output[0][:, j, :] = max((1 - coff), 0) * output[0][:, j, :] + coff * Source_hs[hs_index]
                    else:
                        break
                    output = tuple(output)
            return output
        else:
            output = output
            return output
    return replace


def get_token_layer_positions(mt: ModelAndTokenizer, Text: str, 
                              entity: str, layers: List[int]
                                    ) -> List[Tuple[int, int]]:
    token_range: Tuple[int,int] = get_tokens_range(mt, Text, entity)
    print(f"Token range: {token_range}")
    positions: List[Tuple[int, int]] = []
    for token in range(token_range[0], token_range[1] + 1):
        for layer in range(layers[0], layers[1]):
            positions.append((token, layer))
    filtered_list_for_extract: List[Tuple[int,int]] = [tuple for tuple in positions if tuple[0] == token_range[1]]
    return filtered_list_for_extract

