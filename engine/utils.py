import os
import json
import pickle
import time
import random
from PIL import Image
import openai
import numpy as np
import copy
import collections

import torch
import torch.nn.functional as F

from .step_interpreters import register_step_interpreters, parse_step

random.seed(42)

def openai_api_azure(messages: list, temperature=1.0, top_p=0.5, max_tokens=512):
    
    # raise ValueError()
    
    openai.api_key = 'xxx'
    openai.api_base = 'xxx'
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'
    
    # while True:
    #     try: 
    #         response = openai.Completion.create(
    #             engine="gpt-35-turbo",
    #             prompt=messages[0]['content'],
    #             temperature=temperature,
    #             # top_p=top_p,
    #             # max_tokens=max_tokens,
    #         )
    #         time.sleep(1)
    #         break
    #     except:
    #         time.sleep(random.randrange(5, 10))
    
    response = openai.Completion.create(
        engine="xxx", # nlco-gpt-35-turbo
        prompt=messages[0]['content'],
        temperature=temperature,
        # top_p=top_p,
        max_tokens=max_tokens,
    )
    response = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    
    return response

def openai_api_stream(messages: list, temperature=1.4, top_p=0.5, max_tokens=512):
    
    openai.api_key = ''
    openai.base_url = ''
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=temperature, # 0-2 default 1; larger with more diversity
        # top_p=0.5, # https://huggingface.co/blog/how-to-generate
        # max_tokens=max_tokens,
        stream=True,
    )
    completion = {'role': '', 'content': ''}
    for event in response:
        if event['choices'][0]['finish_reason'] == 'stop':
            # print(f'收到的完成数据: {completion}')
            break
        for delta_k, delta_v in event['choices'][0]['delta'].items():
            # print(f'流响应数据: {delta_k} = {delta_v}')
            completion[delta_k] += delta_v
    messages.append(completion)  # 直接在传入参数 messages 中追加消息
    responses = []
    for dct in messages:
        if dct['role'] == 'assistant':
            responses.append(dct['content'])
    return ''.join(responses)



# to save money, cache the result from openai api ;)
"""
input: prompt, args(model, temperature, top_p)
output: response
"""
def openai_api_generate(prompt, query, index, model='gpt-3.5-turbo', openai_cache_file='cache_openai.pkl', temperature=0.8, top_p=0.5, max_tokens=512):
    
    # openai_cache_file = 'aug_refcoco_cache_openai.pkl'
    modified = 0
    
    if not os.path.exists(openai_cache_file):
        with open(openai_cache_file, 'wb') as pp:
            pickle.dump({}, pp)
    with open(openai_cache_file, 'rb') as pp:
        cached_openai_dct = pickle.load(pp)
    if model not in cached_openai_dct: cached_openai_dct[model] = collections.defaultdict(list)
    
    if model == 'text-davinci-003':
        if query not in cached_openai_dct[model]:
            # # support models: ada, babbage, curie, text-davinci
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=temperature, # 0-1
                max_tokens=512,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                logprobs=0
            )
            response = response.choices[0]['text'].lstrip('\n').rstrip('\n')
            # cached_openai_dct[model][query] = response
            cached_openai_dct[model][query].append(response)
            modified = 1
        else:
            response = cached_openai_dct[model][query][index] # default all demonstration
    
    elif model == 'llama2-chat-13b':
        if query not in cached_openai_dct[model]:
            # # support models: ada, babbage, curie, text-davinci
            response = openai.Completion.create(
                model="/home/wangyuxuan1/prev_trained_models/Llama-2-13b-chat-hf",
                prompt=prompt,
                temperature=temperature, # 0-1
                max_tokens=512,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                logprobs=0
            )
            # print(response)
            response = response.choices[0]['text'].split('\n\n')[0].lstrip('\n').rstrip('\n').strip()
            # cached_openai_dct[model][query] = response
            # print(response)
            cached_openai_dct[model][query].append(response)
            modified = 1
        else:
            response = cached_openai_dct[model][query][index] # default all demonstration
        
    elif model == 'gpt-3.5-turbo':
        # # support models: gpt-3.5-turbo, gpt-3.5-turbo-0301, gpt-3.5-turbo-0613, gpt4
        
        if index == -2:
            messages=[
                {'role': 'user', 'content':prompt}
            ]
            while True:
                try:
                    response = openai_api_stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip('\n')
                    # try:
                    #     response = openai_api_stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip('\n')
                    # except:
                    #     response = openai_api_azure(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()
                    break
                except:
                    time.sleep(random.randrange(5, 10))
            
        else:
        
            if query not in cached_openai_dct[model]:
                messages=[
                        {"role": "user", "content": prompt}
                    ]
                while True:
                    try:
                        response = openai_api_azure(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()
                        # try:
                        #     response = openai_api_stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip('\n')
                        # except:
                        #     response = openai_api_azure(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()
                        break
                    except:
                        time.sleep(random.randrange(5, 10))

                cached_openai_dct[model][query].append(response)
                modified = 1
            else:
                # response = cached_openai_dct[model][query]
                if index >= len(cached_openai_dct[model][query]):
                    index = -1
                response = cached_openai_dct[model][query][index] # default full demonstration
            if modified:
                with open(openai_cache_file, 'wb') as pp:
                    pickle.dump(cached_openai_dct, pp)

        
    return response


def openaiazure_api_generate(prompt, query, index, model='gpt-4', openai_cache_file='cache_openai_gpt4.pkl', temperature=0.8, top_p=0.5, max_tokens=512):
    
    # openai_cache_file = 'aug_refcoco_cache_openai.pkl'
    modified = 0
    
    if not os.path.exists(openai_cache_file):
        with open(openai_cache_file, 'wb') as pp:
            pickle.dump({}, pp)
    with open(openai_cache_file, 'rb') as pp:
        cached_openai_dct = pickle.load(pp)
    if model not in cached_openai_dct: cached_openai_dct[model] = collections.defaultdict(list)
     
    if index == -2:
        messages=[
            {'role': 'user', 'content':prompt}
        ]
        while True:
            try:
                response = openai_api_azure(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()
                break
            except:
                time.sleep(random.randrange(5, 10))
        
    else:
    
        if query not in cached_openai_dct[model]:
            messages=[
                    {"role": "user", "content": prompt}
                ]
            while True:
                try:
                    response = openai_api_azure(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens).strip()
                    break
                except Exception as e:
                    print(e)
                    time.sleep(random.randrange(5, 10))
            
            cached_openai_dct[model][query].append(response)
            modified = 1
        else:
            # response = cached_openai_dct[model][query]
            if index >= len(cached_openai_dct[model][query]):
                index = -1
            response = cached_openai_dct[model][query][index] # default full demonstration
        if modified:
            with open(openai_cache_file, 'wb') as pp:
                pickle.dump(cached_openai_dct, pp)

    
    return response
  

        
class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        # print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        # init_state[dictionary]
        # program : prog_str, state, instructions
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps: # program.prog_str program.state program.instructions
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                # original
                step_output = self.execute_step(prog_step,inspect)
                
            # print(prog_step.prog_str)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state
    
    
    def aug_execute(self,initial_prompts,init_state,question,pre_instruct,prompt_examples,inspect=False,task=''):
        
        openai_model = 'text-davinci-003'
        openai_model = 'gpt-3.5-turbo'
        openai_model = 'llama2-chat-13b_aug'
        # openai_cache = openai_model + '_' + self.dataset + '_aug.pkl'
        openai_cache = openai_model + '_' + task + '.pkl'
        
        # init_state[dictionary]
        # [program : prog_str, state, instructions]
        k = 2 # rollout
        beam_size = 2 # beam_szie
        constrained_beam_size = 4 # constrained beam size for causal
        max_step = 8
        step = 0
        if task == 'gqa':
            neural_modules = ['VQA', 'LOC']
            neural_modules = ['LOC']
        elif task == 'refcoco':
            neural_modules = ['LOC']
        elif task == 'kilogram':
            neural_modules = ['']
        elif task == 'magicbrush':
            neural_modules = ['SELECT']
        elif task == 'nlvr':
            neural_modules = ['VQA']
        elif task == 'agqa':
            neural_modules = ["LOC"]
        
        fail_cases = set()
        retry_cnt = 0
        if task == 'nlvr': retry_cnt = 2
        loop_cnt = 0
        
        # we start with some different initialized prompter
        states = []
        for i in range(len(initial_prompts)):
            # prog = openai_api_generate(initial_prompts[i] + initial_instruct.format(**dict(question=question, rejected_solutions='')), initial_prompts[i]+initial_instruct.format(**dict(question=question,rejected_solutions='')), -1, temperature=0.7).strip()
            pre_inst = pre_instruct.format(**dict(rejected_solutions=''))
            
            # prog = openai_api_generate(pre_inst + initial_prompts[i], pre_inst + initial_prompts[i], -1, temperature=0.7, openai_cache_file=task+'openai_cache.pkl').strip()
            prog = openai_api_generate(pre_inst + initial_prompts[i], pre_inst + initial_prompts[i], -1, temperature=0.7, openai_cache_file=openai_cache).strip()
            if 'Program' in prog: continue
            # if 'Program' in prog: raise ValueError()
            prog = Program(prog, init_state)
            # prompt, path, prog, ct_score, cd_score, result, complete_flag
            # 0        1      2      3        4        5        6
            #-7       -6       -5     -4     -3       -2      -1       
            states.append([initial_prompts[i], '', prog, 0, 0, 'NA', 0])
            # states.append([initial_prompts[i], '', prog, [0], 'NA', 0])
            
        while True:
            
            loop_cnt += 1
            
            # excute and beam constrain
            new_states = []
            for idx in range(len(states)):
                # prog = openai_api_generate(initial_prompt, initial_prompt, -1)
                # prog = Program(prog, init_state)
                
                state = states[idx]
                if state[-1]: 
                    new_states.append(states[idx])
                    continue
                
                if len(state[1].split('\n')) > max_step:
                    state[-1] = 0
                    state[3] = 0
                    state[4] = 0
                    # new_states.append(state)
                    continue
                
                prog = state[2]
                
                prog_steps = []
                for instruction in prog.instructions:
                    if instruction and 'Program' not in instruction:
                        prog_steps.append(Program(instruction,init_state=prog.state))
                    else:
                        state[1] = ''
                        state[3] = 0
                        state[4] = 0
                        continue
                        

                # prog_steps = [Program(instruction,init_state=prog.state) \
                #              for instruction in prog.instructions]

                complete = 0
                error = 0
                # init step
                for prog_step in prog_steps:
                    # if current module is neural module
                    step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
                    if step_name not in neural_modules:
                        try:
                            states[idx][1] += ('\n' + prog_step.prog_str)
                            
                            # # evaluate causality discovery
                            # path = states[idx][1]
                            # cd_prompt = f""" You are a rater for a planner who use the candidate modules include LOC: detect object, VQA: visual question answering, EVAL: evaluate, RESULT: wrap result, COUNT: count object, CROP: crop image, , CROP_RIGHTOF, CROP_LEFT, CROP_FRONTOF, CROP_INFRONTOF, CROP_INFRONT, CROP_BEHIND, CROP_AHEAD, CROP_BELOW, CROP_ABOVE.
                            # To answer the question: '{question}', pessimistically value the current solutions for answering the question you had AS A FLOAT BETWEEN 0 AND 1\n
                            # current solutions:\n\n
                            # {path}\n       
                            # Considering the relationships about cause and effect of different modules. 
                            # Evaluate current solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                            # """
                            # response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
                            # try:
                            #     value = float(response)
                            # except:
                            #     value = 0
                            # states[idx][4] = value
                            
                            # print(states[idx][1])
                            # print()
                            step_output = self.execute_step(prog_step, inspect)
                            # states[idx][0] += ('\n' + prog_step.prog_str)
                        except Exception as e:
                            e
                            error = 1
                            break
                        
                    else:
                        break
                    if step_name == 'RESULT':
                        complete = 1
                
                if error:
                    states[idx][-1] = 0
                    states[idx][3] = 0
                    states[idx][4] = 0
                    fail_cases.add(states[idx][1])
                    # # new_prog = openai_api_generate(states[idx][0] + initial_instruct.format(**dict(question=question, rejected_solutions=states[idx][1])), states[idx][0] + initial_instruct.format(**dict(question=question,rejected_solutions=states[idx][1])), -1, temperature=0.7).strip()
                    # pre_inst = pre_instruct.format(**dict(rejected_solutions=states[idx][1]))
                    # new_prog = openai_api_generate(pre_inst + states[idx][0], pre_inst + states[idx][0], -2, temperature=0.5).strip()
                    # new_prog = Program(new_prog, init_state=init_state)
                    # new_states.append([states[idx][0], '', new_prog, 0, 'NA', 0])
                    continue
        

                if complete:
                    # states[idx][-2] = step_output
                    if task == 'gqa' or task == 'magicbrush' or task == 'kilogram' or task == 'nlvr' or task == 'agqa':
                        states[idx][-2] = step_output
                    elif task == 'refcoco':
                        states[idx][-2] = prog_step.state['BOX'][0]
                    states[idx][-1] = 1
                    # states[idx][1] = prog
                    new_states.append(states[idx])
                else:
                    try:
                        states[idx][1] += ('\n' + prog_step.prog_str)
                        # print(states[idx][1])
                        # print()
                        
                        # # evaluate causality discovery
                        # path = states[idx][1]
                        # cd_prompt = f""" You are a rater for a planner who use the candidate modules include LOC: detect object, VQA: visual question answering, EVAL: evaluate, RESULT: wrap result, COUNT: count object, CROP: crop image, , CROP_RIGHTOF, CROP_LEFT, CROP_FRONTOF, CROP_INFRONTOF, CROP_INFRONT, CROP_BEHIND, CROP_AHEAD, CROP_BELOW, CROP_ABOVE.
                        # To answer the question: '{question}', pessimistically value the current solutions for answering the question you had AS A FLOAT BETWEEN 0 AND 1\n
                        # current solutions:\n\n
                        # {path}\n       
                        # Considering the relationships about cause and effect of different modules. 
                        # Evaluate current solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                        # """
                        # response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
                        # try:
                        #     value = float(response)
                        # except:
                        #     value = 0
                        # states[idx][4] = value
                        
                        
                        step_output = self.execute_step(prog_step, inspect)
                        # states[idx][0] += ('\n' + prog_step.prog_str)
                    except:
                        states[idx][-1] = 0
                        states[idx][3] = 0
                        states[idx][4] = 0
                        fail_cases.add(states[idx][1])
                        # pre_inst = pre_instruct.format(**dict(rejected_solutions=states[idx][1]))
                        # new_prog = openai_api_generate(pre_inst + states[idx][0], pre_inst + states[idx][0], -2, temperature=0.5).strip()
                        # # new_prog = openai_api_generate(states[idx][0] + initial_instruct.format(**dict(question=question, rejected_solutions=states[idx][1])), states[idx][0] + initial_instruct.format(**dict(question=question,rejected_solutions=states[idx][1])), -1, temperature=0.7).strip()
                        # new_prog = Program(new_prog, init_state=init_state)
                        # new_states.append([states[idx][0], '', new_prog, 0, 'NA', 0])
                        continue
                    # states[idx][2] = prog
                    states[idx][3] += prog.state['CT_SCORE']
                    # states[idx][4] += 1
                    new_states.append(states[idx])
            
            # beam constrain
            states = new_states
            # states = sorted(states, key=lambda x:x[3]/x[4], reverse=True)
            # states = sorted(states, key=lambda x:x[-1], reverse=True) # get complete results
            states = sorted(states, key=lambda x:x[3], reverse=True)
            
            states = states[:constrained_beam_size]
            
            if len(states) > 1:
                if task == 'gqa':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to answer a question: {question}, select the best solutions for answering the question\n cadidate modules include: LOC: detection, VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result, CROP/CROP_LEFTOF/CROP_RIGHTOF/CROP_FRONTOF/CROP_INFRONT/CROP_INFRONTOF/CROP_BEHIND/CROP_AHEAD/CROP_BELOW/CROP_ABOVE: crop the image. \n Current solutions:\n\n"""
                elif task == 'refcoco':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to carry out the instruction: {question}, select the best solutions for carrying out the instruction\n cadidate modules include: LOC: detection, FILTER: filter unrelated objects, TAG: tag the object, RESULT: wrap up the final result \n Current solutions:\n\n"""
                elif task == 'magicbrush':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to carry out the instruction: {question}, select the best solutions for carrying out the instruciton\n cadidate modules include: SEG: segmentation, SELECT: select most related object, REPALCE: edit image, RESULT: wrap up the final result \n Current solutions:\n\n"""
                elif task == 'kilogram':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to carry out the instruction: {question}, select the best solutions for carrying out the instruciton\n cadidate modules include: PART: take apart an object, SEG: segment, ALIGN: align object with query, RESULT: wrap up the final result \n Current solutions:\n\n"""
                elif task == 'nlvr':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to evaluate the statement: {question}, select the best solutions for evaluating the statement\n cadidate modules include: VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result \n Current solutions:\n\n"""
                elif task == 'agqa':
                    cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to answer a question: {question}, select the best solutions for answering the question\n cadidate modules include: FIND: video temporal grounding, VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result, MEASURE: get temporal length, GET/GET_BEFORE/GET_AFTER/GET_BETWEEN: clip video, LOC: detection. , CROP/CROP_LEFTOF/CROP_RIGHTOF/CROP_FRONTOF/CROP_INFRONT/CROP_INFRONTOF/CROP_BEHIND/CROP_AHEAD/CROP_BELOW/CROP_ABOVE: crop the image. \n Current solutions:\n\n"""
                
                cd_prompt_mid = ""
                for ii in range(len(states)):
                    if task == 'gqa' or task == 'agqa':
                        cd_prompt_mid += str(ii) + ' \n ' + f"question: {question} \n {states[ii][1]} \n"
                    elif task == 'refcoco' or task == 'magicbrush' or task == 'kilogram':
                        cd_prompt_mid += str(ii) + ' \n ' + f"instruction: {question} \n {states[ii][1]} \n"
                    elif task == 'nlvr':
                        cd_prompt_mid += str(ii) + ' \n ' + f"statement: {question} \n {states[ii][1]} \n"
                
                cd_prompt_post = """If the modules in the solutions have better cause-and-effect relations, and more likely to answer the question, please rank it first. If you are unsure, please keep the original rank. Return sequence number of currently best solution, for example 0,1,2,3 \n, DO NOT RETURN ANYTHING ELSE EXCEPT FOR NUMBERS SPLIT by ,"""
                cd_prompt = cd_prompt_prev + cd_prompt_mid + cd_prompt_post
                # response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10,openai_cache_file=task+'openai_cache.pkl').strip()
                response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10,openai_cache_file=openai_cache).strip()
                response = response.replace(' ', '')
                try:
                    indexs = response.split(',')
                    indexs = [int(x) for x in indexs if int(x) < len(states)]
                except:
                    indexs = list(range(len(states)))
                states = [states[x] for x in indexs]
            
                # # rank-augmented score
                # rank_scores = []
                # lb = 0.5
                # for ii in range(len(states)):
                #     if ii not in indexs:
                #         rank_scores.append(1)
                #     else:
                #         rank_scores.append(beam_size*k-indexs.index(ii))
                # minx, maxx = min(rank_scores), max(rank_scores)
                # normed_rank_scores = [(x-minx)/(maxx-minx) * (1-lb) + lb for x in rank_scores]
                # for ii in range(len(states)):
                #     states[ii][3] *= normed_rank_scores[ii]
            
            # states = sorted(states, key=lambda x:x[3], reverse=True)
            
            
            states = states[:beam_size] 
            
            # # # way1 evaluate value
            # # if len(states) > 1:
            # for state in states:
            #     # # evaluate causality discovery
            #     path = state[1]
            #     cd_prompt = f""" You are a evaluator for a planner who use the candidate modules include LOC: detect object, VQA: visual question answering, EVAL: evaluate, RESULT: wrap result, COUNT: count object, CROP: crop image, , CROP_RIGHTOF, CROP_LEFT, CROP_FRONTOF, CROP_INFRONTOF, CROP_INFRONT, CROP_BEHIND, CROP_AHEAD, CROP_BELOW, CROP_ABOVE.
            #     To answer the question: '{question}', pessimistically value the current solutions for answering the question you had AS A FLOAT BETWEEN 0 AND 1\n
            #     current solutions:\n\n
            #     {path}\n       
            #     If the solutions is not making modules for answering the question or have bad cause-and-effect relations, give it a lower score.
            #     Evaluate current solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
            #     """
            #     response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
            #     try:
            #         value = float(response)
            #     except:
            #         value = 0
            #     state[4] = value
            # states = sorted(states, key=lambda x:x[4], reverse=True)
            
            #  # # way2 vote
            # if len(states) > 1:
            #     cd_prompt_prev = f""" You are a ranker for a planner who use the candidate modules to answer a question: {question}, select the best solutions for answering the question\n cadidate modules include: LOC: detection, VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result, CROP/CROP_LEFTOF/CROP_RIGHTOF/CROP_FRONTOF/CROP_INFRONT/CROP_INFRONTOF/CROP_BEHIND/CROP_AHEAD/CROP_BELOW/CROP_ABOVE: crop the image. \n Current solutions:\n\n"""
                
            #     cd_prompt_mid = ""
            #     for ii in len(range(states)):
            #         cd_prompt_mid += str(ii) + ' \n ' + f"question: {question} \n {states[ii][1]} \n"
                
            #     cd_prompt_post = """If the modules in the solutions have better cause-and-effect relations, and more likely to answer the question, please rank it first. If you are unsure, please keep the original rank. Return sequence number of currently best solution, for example 0, 1, 2, 3\n, DO NOT RETURN ANYTHING ELSE EXCEPT FOR NUMBER"""
            #     cd_prompt = cd_prompt_prev + cd_prompt_mid + cd_prompt_post
            #     response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
            #     try:
            #         indexs = response.split(',')
            #         indexs = [int(x) for x in indexs if int(x) < len(states)]
            #     except:
            #         index = 0
            #     states = [states[x] for x in indexs]
            
            # # way2 vote
            # if all_complete and len(states) > 1:
            #     cd_prompt = f""" You are a voter for a planner who use the candidate modules to answer a question, select the best solutions for answering the question\n
            #     cadidate modules include: LOC: detection, VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result, CROP/CROP_LEFTOF/CROP_RIGHTOF/CROP_FRONTOF/CROP_INFRONT/CROP_INFRONTOF/CROP_BEHIND/CROP_AHEAD/CROP_BELOW/CROP_ABOVE
            #     here are examples of the planner result:
            #     question: Does the traffic cone have white color? \n
            #     BOX0=LOC(image=IMAGE,object='traffic cone') \n
            #     IMAGE0=CROP(image=IMAGE,box=BOX0) \n
            #     ANSWER0=VQA(image=IMAGE0,question='What color is the traffic cone?') \n
            #     ANSWER1=EVAL(expr="'yes' if ANSWER0 == 'white' else 'no'") \n
            #     FINAL_RESULT=RESULT(var=ANSWER1) \n
            #     question: Do you see any drawers to the left of the plate? \n
            #     BOX0=LOC(image=IMAGE,object='plate') \n
            #     ANSWER0=COUNT(box=BOX1) \n
            #     IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0) \n
            #     BOX1=LOC(image=IMAGE0,object='drawers') \n
            #     ANSWER1=COUNT(box=BOX1) \n
            #     ANSWER2=EVAL(expr="'yes' if ANSWER0 > 0 and ANSWER1 > 0 else 'no'") \n
            #     FINAL_RESULT=RESULT(var=ANSWER2) \n
            #     current solutions:\n\n
            #     0 \n question: {question} \n {states[0][1]}\n       
            #     1 \n question: {question} \n {states[1][1]}\n
            #     If the solutions have better cause-and-effect relations, and more likely to answer the question, please chose it. If you are uncertain, please choose 0.
            #     Select sequence number of currently best solution, for example 0\n, DO NOT RETURN ANYTHING ELSE EXCEPT FOR NUMBER
            #     """
            #     response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
            #     try:
            #         index = int(response)
            #         assert index < len(states) and index >= 0
            #     except:
            #         index = 0
            #     states[0], states[index] = states[index], states[0]
            
            
            # if new_states:
            #     # beam constrain
            #     states = new_states
            #     # states = sorted(states, key=lambda x:x[3]/x[4], reverse=True)
            #     eval_states = copy.deepcopy(states)
            #     n_cd_score = []
            #     # for eval_state in eval_states:
            #     #     # # evaluate causality discovery
            #     #     path = eval_state[1]
            #     #     cd_prompt = f""" You are a rater for a planner who use the candidate modules include LOC: detect object, VQA: visual question answering, EVAL: evaluate, RESULT: wrap result, COUNT: count object, CROP: crop image, , CROP_RIGHTOF, CROP_LEFT, CROP_FRONTOF, CROP_INFRONTOF, CROP_INFRONT, CROP_BEHIND, CROP_AHEAD, CROP_BELOW, CROP_ABOVE.
            #     #     To answer the question: '{question}', pessimistically value the current solutions for answering the question you had AS A FLOAT BETWEEN 0 AND 1\n
            #     #     current solutions:\n\n
            #     #     {path}\n       
            #     #     If the solutions is not making modules for answering the question or have bad causal relations, give it a lower score.
            #     #     Evaluate current solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
            #     #     """
            #     #     response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
            #     #     try:
            #     #         value = float(response)
            #     #     except:
            #     #         value = 0
            #     #     n_cd_score.append(value)
                
                
            #     n_ct_score = [x[3] for x in eval_states]
            #     # n_score = (F.normalize(torch.tensor(n_ct_score, dtype=float), dim=-1) * 0.8+ F.normalize(torch.tensor(n_cd_score, dtype=float), dim=-1) * 0.2).tolist()
            #     n_score = n_ct_score
            #     # n_score = n_cd_score
            #     states, _ = zip(*sorted(zip(states, n_score), key=lambda x:x[1], reverse=True))
                    
            #     # states = sorted(states, key=lambda x:x[3], reverse=True)
            #     # states = sorted(states, key=lambda x:x[-1], reverse=True) # get complete results
            #     states = states[:beam_size]
            #     # print(states)
                
            #     # if loop_cnt > 2:
            #     #     states = [state for state in states if state[-1]]
                             
            # check stop condition
            all_complete = 1
            for state in states:
                if state[-1] == 0:
                    all_complete = 0
                    break
            if all_complete: 
                
                # # way2 vote
                # if len(states) > 1:
                #     cd_prompt = f""" You are a selector for a planner who use the candidate modules to answer a question: {question}, select the best solutions for answering the question\n
                #     cadidate modules include: LOC: detection, VQA: visual question answering, EVAL: use logic operation, RESULT: wrap up the final result, CROP/CROP_LEFTOF/CROP_RIGHTOF/CROP_FRONTOF/CROP_INFRONT/CROP_INFRONTOF/CROP_BEHIND/CROP_AHEAD/CROP_BELOW/CROP_ABOVE: crop the image
                #     current solutions:\n\n
                #     0 \n question: {question} \n {states[0][1]}\n       
                #     1 \n question: {question} \n {states[1][1]}\n
                #     If the modules in the solutions have better cause-and-effect relations, and more likely to answer the question, please choose it. If you are uncertain, please choose 0.
                #     Select sequence number of currently best solution, for example 0\n, DO NOT RETURN ANYTHING ELSE EXCEPT FOR NUMBER
                #     """
                #     response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
                #     try:
                #         index = int(response)
                #         assert index < len(states) and index >= 0
                #     except:
                #         index = 0
                #     if index == 1:
                #         print()
                #     states[0], states[index] = states[index], states[0]
                
                
                if states == []:
                    retry_cnt += 1
                    pre_inst = pre_instruct.format(**dict(rejected_solutions='\n'.join(fail_cases)))
                    init_prompt = random.sample(prompt_examples, int(len(prompt_examples) * random.uniform(0.4, 1.0)))
                    if task == 'gqa' or task == 'agqa':
                        init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nQuestion: {question}\nProgram:"
                    elif task == 'refcoco' or task == 'magicbrush' or task == 'kilogram':
                        init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nInstruction: {question}\nProgram:"
                    elif task == 'nlvr':
                        init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nStatement: {question}\nProgram:"
                    
                    if retry_cnt > 2:
                        # new_prog = openai_api_generate(pre_inst + init_prompt, pre_inst + init_prompt, -1, temperature=0.7,openai_cache_file=task+'openai_cache.pkl').strip()
                        new_prog = openai_api_generate(pre_inst + init_prompt, pre_inst + init_prompt, -1, temperature=0.7,openai_cache_file=openai_cache).strip()
                        new_prog = Program(new_prog, init_state)
                        step_output = 'NA'
                        # new_prog_steps = [Program(instruction,init_state=new_prog.state) \
                        #      for instruction in new_prog.instructions if instruction and 'Program' not in instruction]
                        new_prog_steps = [Program(instruction,init_state=new_prog.state) \
                             for instruction in new_prog.instructions if instruction and 'Program' not in instruction]
                        for new_prog_step in new_prog_steps:
                            try:
                                step_output = self.execute_step(new_prog_step, inspect)
                                if task == 'refcoco':
                                    step_output = new_prog_step.state['BOX'][0]
                                # print(new_prog_step.prog_str)
                            except:
                                # states.append([init_prompt, '', new_prog, 0, 0, 'NA', 1])
                                break
                        states.append([init_prompt, '', new_prog, 0, 0, step_output, 1])
                        break
                            
                        
                    else:
                        
                        # new_prog = openai_api_generate(pre_inst + init_prompt, pre_inst + init_prompt, -1, temperature=0.7,openai_cache_file=task+'openai_cache.pkl').strip()
                        new_prog = openai_api_generate(pre_inst + init_prompt, pre_inst + init_prompt, -1, temperature=0.7,openai_cache_file=openai_cache).strip()
                        new_prog = Program(new_prog, init_state)
                        states.append([init_prompt, '', new_prog, 0, 0, 'NA', 0])
                        
                else:                
                    break
            
                
            
            # extend
            new_states = []
            for idx in range(len(states)):
                             
                state = states[idx]
                if state[-1]: 
                    new_states.append(states[idx])
                    continue
                
                prog = state[2]

                prog_steps = []
                for instruction in prog.instructions:
                    if instruction and 'Program' not in instruction:
                        prog_steps.append(Program(instruction,init_state=prog.state))
                    else:
                        state[1] = ''
                        state[3] = 0
                        state[4] = 0
                        continue
                
                # prog_steps = [Program(instruction,init_state=prog.state) \
                #              for instruction in prog.instructions if 'Program' not in instruction]

                complete = 0
                error = 0
                # init step
                for prog_step in prog_steps:
                    # if current module is neural module
                    step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
                    # states[idx][1] += ('\n' + prog_step.prog_str)
                    if step_name not in neural_modules:
                        try:
                            states[idx][1] += ('\n' + prog_step.prog_str)
                            # print(states[idx][1])
                            # print()
                            
                            # # evaluate causality discovery
                            # path = states[idx][1]
                            # cd_prompt = f""" You are a rater for a planner who use the candidate modules include LOC: detect object, VQA: visual question answering, EVAL: evaluate, RESULT: wrap result, COUNT: count object, CROP: crop image, , CROP_RIGHTOF, CROP_LEFT, CROP_FRONTOF, CROP_INFRONTOF, CROP_INFRONT, CROP_BEHIND, CROP_AHEAD, CROP_BELOW, CROP_ABOVE.
                            # To answer the question: '{question}', pessimistically value the current solutions for answering the question you had AS A FLOAT BETWEEN 0 AND 1\n
                            # current solutions:\n\n
                            # {path}\n       
                            # Considering the relationships about cause and effect of different modules. 
                            # Evaluate current solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                            # """
                            # response = openai_api_generate(cd_prompt, cd_prompt, -1, temperature=1.0, max_tokens=10).strip()
                            # try:
                            #     value = float(response)
                            # except:
                            #     value = 0
                            # states[idx][4] = value
                            
                            step_output = self.execute_step(prog_step, inspect)
                        except Exception as e:
                            e
                            error = 1
                            break
                    else:
                        break
                    if step_name == "RESULT":
                        complete = 1
                if error:
                    states[idx][-1] = 0
                    states[idx][3] = 0
                    states[idx][4] = 0
                    fail_cases.add(states[idx][1])
                    # # new_prog = openai_api_generate(states[idx][0] + initial_instruct.format(**dict(question=question, rejected_solutions=states[idx][1])), states[idx][0] + initial_instruct.format(**dict(question=question,rejected_solutions=states[idx][1])), -1, temperature=0.7).strip()
                    # pre_inst = pre_instruct.format(**dict(rejected_solutions=states[idx][1]))
                    # new_prog = openai_api_generate(pre_inst + states[idx][0], pre_inst + states[idx][0], -2, temperature=0.5).strip()
                    # new_prog = Program(new_prog, init_state=init_state)
                    # new_states.append([states[idx][0], '', new_prog, 0, 'NA', 0])
                    continue
                
                if complete:
                    if task == 'gqa' or task == 'magicbrush' or task == 'kilogram' or task == 'nlvr' or task == 'agqa':
                        states[idx][-2] = step_output
                    elif task == 'refcoco':
                        states[idx][-2] = prog_step.state['BOX'][0]
                    states[idx][-1] = 1
                    # states[idx][2] = prog
                    new_states.append(states[idx])
                else:
                    cur_ct_score = states[idx][3] + prog.state['CT_SCORE']
                    cur_cd_score = states[idx][4]
                    # cur_prompt = states[idx][0] + initial_instruct.format(**dict(question=question, rejected_solutions='')) + states[idx][1]
                    pre_inst = pre_instruct.format(**dict(rejected_solutions=''))
                    cur_prompt = pre_inst + states[idx][0] + states[idx][1]
                    cur_path = states[idx][1]
                    # states[idx][0] = cur_prompt
                    # states[idx][-4] = prog
                    # extend to k
                    for _ in range(k):
                        init_prompt = random.sample(prompt_examples, int(len(prompt_examples) * random.uniform(0.7,1.0)))
                        if task == 'gqa' or task == 'agqa':
                            init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nQuestion: {question}\nProgram:"
                        elif task == 'refcoco' or task == 'magicbrush' or task == 'kilogram':
                            init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nInstruction: {question}\nProgram:"
                        elif task == 'nlvr':
                            init_prompt = "Considering the examples provided:\n\n" + '\n'.join(init_prompt) + f"\nStatement: {question}\nProgram:"
                        cur_prompt = pre_inst + init_prompt + cur_path
                        cur_prog = openai_api_generate(cur_prompt, cur_prompt, -1, temperature=0.5,openai_cache_file=task+'openai_cache.pkl').strip()
                        if 'Program' in cur_prog: continue
                        # if 'Program' in cur_prog: raise ValueError()
                        cur_prog = Program(cur_prog, prog.state)
                        # if cur_prog.instructions[0] == cur_prog
                        new_states.append([states[idx][0], cur_path, cur_prog, cur_ct_score, cur_cd_score, 'NA', 0])
            states = new_states
             
        # print('################## PATH #####################')
        if states:
            # print(states)
            _, path, prog, ct_score, cd_score, result, _ = states[0]
            if result == 'NA':
                print(path)
            # print(path)
            # print('#######################################')
            return result, ct_score, cd_score
        else:
            # print('#######################################')
            return 'NA', 0, 0
        


class ProgramGenerator():
    def __init__(self,prompter,dataset,temperature=0.7,top_p=0.5,prob_agg='mean'):
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        self.prompter = prompter
        # self.answer = prompter.split('?')[-1]
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        self.dataset = dataset

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def generate(self,inputs,index=-2):
        
        openai_model = 'text-davinci-003'
        openai_model = 'gpt-3.5-turbo'
        # openai_model = 'llama2-chat-13b'
        # openai_cache = openai_model + '_' + self.dataset + '_aug.pkl'
        openai_cache = openai_model + '_' + self.dataset + '_mmbench.pkl'
        # openai_cache = 'baseline_cache_openai.pkl'
        
        if 'question' in inputs:
            query = inputs['question']
        elif 'statement' in inputs:
            query = inputs['statement']
        elif 'instruction' in inputs:
            query = inputs['instruction']
        else:
            query = inputs
        
        prog = openai_api_generate(self.prompter(inputs), query, index, openai_model, openai_cache)
        prob = None
        
        return prog, prob
                             
    def generate_prompt(self,inputs):
        
        return self.prompter(inputs)
    