import cv2
import os
import time
import random
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
import pickle
from collections import OrderedDict
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter

from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPConfig, CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering,
    Blip2Processor, Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration)
from diffusers import StableDiffusionInpaintPipeline
from transformers.models.bert.modeling_bert import BertAttention
import torch.nn as nn
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from sentence_transformers import SentenceTransformer, util

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import inflect
inflect_engine = inflect.engine()
from trie import MarisaTrie

import json
from pathlib import Path
anto_file = os.path.join(Path.home(), "codes/visjoint/datasets/english_vocabulary/antos.json")
with open(anto_file) as jh:
    anto_dct = json.load(jh)
ckpt_dir = 'prev_trained_models'

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

CUDAIDX = os.environ.get('CUDAIDX', 'cuda:0')
# CUDAIDX = 'cuda:2'

def openai_api_stream(messages: list):
    
    retry_cnt = 0
    while 1:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=1.0, # 0-2 default 1; larger with more diversity
                top_p=0.5, # https://huggingface.co/blog/how-to-generate
                stream=True,
            )
            break
        except Exception as e:
            retry_cnt += 1
            print(e)
            print(f'start retry {retry_cnt}')
            time.sleep(random.randrange(5,30))
            # if retry_cnt > 20: raise Exception
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
    return '\n'.join(responses)


def parse_step(step_str,partial=False):
    # print(step_str)
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    # print(tokens)
    # BOX1 = LOC ( image = IMAGE , object = 'animal' )
    #   0  1  2  3   4   5   6   7   8    9     10  11  12  13
    output_var = tokens[0].string # BOX2
    step_name = tokens[2].string # LOC
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result
    # image IMAGE object 'animal'
    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    """
    {
        "output_var": "BOX2",
        "step_name": "LOC",
        "image": "IMAGE",
        "object": "animal",
    }
    """
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'


def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'


class SimInterpreter():
    step_name = 'SIM'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        
        # self.clip_model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        load_path = '/home/patrick/codes/templates/lightning-hydra-template/logs/train/runs/2023-08-04_20-47-11/checkpoints/last.ckpt'
        # config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
        config = CLIPConfig.from_pretrained(f"{ckpt_dir}/clip-vit-large-patch14")
        self.clip_model = CLIPModel(config)
        ckpt = torch.load(load_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if 'model' in k:
                k = k[6:]
            new_state_dict[k] = v
        self.clip_model.load_state_dict(new_state_dict)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b").to(self.device)
        self.blip_model.eval()
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query_var = parse_result['args']['query']
        answer_var = parse_result['args']['answer']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query_var,answer_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.clip_model.get_image_features(inputs['pixel_values'])
        text_feats = self.clip_model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def calculate_sim_text(self,query,captions):
        query_feats = self.clip_model.get_text_features(query)
        caption_feats = self.clip_model.get_text_features(captions)
        query_feats = query_feats / query_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(caption_feats,query_feats.t())

    def cal_sim(self,query,objs,img):
        # print(objs)
        if objs == []:
            return 0, 'NO CAPTION', 0
        if objs != 'none':
            images = [img.crop(obj) for obj in objs]
        else:
            images = [img]
            
        # calculate similarity between images and query
        clip_inputs = self.clip_processor(
            text=f"a photo of a {query}", images=images, return_tensors="pt", padding=True)
        clip_inputs = {k:v.to(self.device) for k,v in clip_inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(clip_inputs).cpu().numpy()
        similarity = scores
        # generate caption
        blip_inputs = self.blip_processor(
            images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            caption_ids = self.blip_model.generate(**blip_inputs)
            captions = self.blip_processor.batch_decode(caption_ids)
        # calculate similarity between captions and query
        caption_inputs = self.clip_processor(text=captions, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            tscores = self.calculate_sim_text(clip_inputs['input_ids'], caption_inputs['input_ids']).cpu().numpy()
        capsimilarity = tscores
        
        return similarity, captions, capsimilarity

    def html(self,img_var,obj_var,query,output_var,similarity,caption,capsimilarity):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        caption_arg = html_arg_name('caption')
        caption_sim_arg = html_var_name('caption similarity')
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{caption_arg}={caption},{caption_sim_arg}={capsimilarity},{query_arg}={query})={similarity}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,query,answer_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_var != 'none' and obj_var != []:
            objs = prog_step.state[obj_var]
        else:
            objs = 'none'

        openai_cache_file = 'cache_openai.json'
        if not os.path.exists(openai_cache_file):
            with open(openai_cache_file, 'w') as jp:
                json.dump({}, jp)
        with open(openai_cache_file) as jp:
            cached_openai_dct = json.load(jp)
        
        if answer_var != 'none':
            answer = prog_step.state[answer_var]
            convert_prompt = f"""Turn the following QA pair to declarative sentence: 
                Question: Are there trains or fences in this scene?
                Answer: no
                Declarative: There aren't trains or fences in this scene
                Question: Who is carrying the umbrella?
                Answer: the man
                Declarative: The man is carrying the umbrella
                Question: Is the pillow in the top part or in the bottom of the picture?
                Answer: top part
                Declarative: The pillow is in the top part of the picture
                Question: Which side is the food on?
                Answer: RIGHT
                Declarative: The food is on right
                Question: What do the wetsuit and the sky have in common?
                Answer: they are blue
                Declarative: Both the wetsuit and the sky are blue
                Question: {query}
                Answer: {answer}
                Declararive: """
            if convert_prompt not in cached_openai_dct:
                query = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", #0301 0613
                    messages=[
                        {"role": "user", "content": convert_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                ).choices[0].message['content'].lstrip('\n').rstrip('\n')
            else:
                query = cached_openai_dct[convert_prompt]
        else:
            query = f"a photo of {query}"

        similarity, caption, capsimilarity = self.cal_sim(query, objs, img)
        prog_step.state[output_var] = similarity
        
        if inspect:
            html_str = self.html(img_var, obj_var, query, output_var, similarity, caption, capsimilarity)
            return similarity, html_str

        return similarity


class FindInterpreter():
    step_name = 'FIND'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(f"{ckpt_dir}/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            f"{ckpt_dir}/blip-image-captioning-large").to(self.device)
        self.model.eval()
        self.sent_model = SentenceTransformer(f'{ckpt_dir}/sentence-transformers_all-MiniLM-L6-v2').to(self.device)
        self.sent_model.eval()

        self.clip_model = CLIPModel.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14"
        )

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        query = eval(args['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,query,output_var

    def predict(self,img, query):
        inputs = self.processor(img,return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        query_embedding = self.sent_model.encode(query, convert_to_tensor=True, device=self.device)
        caption_embedding = self.sent_model.encode(captions, convert_to_tensor=True, device=self.device)
        caption_sims = util.pytorch_cos_sim(caption_embedding, query_embedding).squeeze(-1)        
        # _, _, F1 = score(captions, query, lang='en', baseline_path=f'{ckpt_dir}/roberta-large')
        bs = 0
        start_target = 0
        end_target = len(caption_sims)-1
        stack = []
        caption_sims = caption_sims.tolist()
        caption_sims = [caption_sims[i]-min(caption_sims) for i in range(len(caption_sims))]
        # score = [0] + caption_sims.tolist() + [0]
        score = [0] + caption_sims + [0]
        # print(score)
        for i in range(len(score)):
            while stack and score[stack[-1]] > score[i]:
                tmp = stack.pop()
                tmp_bs = (i-stack[-1]-1) * score[tmp]
                if tmp_bs > bs:
                    bs = tmp_bs
                    start_target, end_target = stack[-1], i-2
            stack.append(i)
        idxs = list(range(start_target, end_target+1))
        return idxs, 0
    
    def calculate_sim(self,inputs):
        img_feats = self.clip_model.get_image_features(inputs['pixel_values'])
        text_feats = self.clip_model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        # sims = torch.diagonal(torch.matmul(text_feats,img_feats.t()))
        sims = torch.sum(img_feats*text_feats, dim=-1)
        return sims

    def cal_sim_aug_score(self, img, querys, ct_querys):
        # querys = ['a photo of ' + query for query in querys]
        # ct_querys = ['a photo of ' + ct_query for ct_query in ct_querys]
        clip_inputs = self.clip_processor(
            text = querys,
            # images=[img]*len(responses),
            images=img,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
            
        ct_clip_inputs = self.clip_processor(
            text=ct_querys,
            # images=[img]*len(ct_responses),
            images=img,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            ct_sims = self.calculate_sim(ct_clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        # counter_sims = sims
                
        return counter_sims
    
    def ct_aug_predict(self,img,query):
        inputs = self.processor(img,return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        query_embedding = self.sent_model.encode(query, convert_to_tensor=True, device=self.device)
        caption_embedding = self.sent_model.encode(responses, convert_to_tensor=True, device=self.device)
        caption_sims = util.pytorch_cos_sim(caption_embedding, query_embedding).squeeze(-1)        
        
        # verify
        querys = [response for response in responses]
        ct_querys = []
        for query in querys:
            tag_query = pos_tag(word_tokenize(query))
            tokens = [x[0] for x in tag_query]
            tags = [x[1] for x in tag_query]
            noun_indices = [ii for ii in range(len(tags)) if 'NN' in tags[ii]]
            if noun_indices == []:
                ct_querys.append('not' + query)
            else:
                noun_indice = random.choice(noun_indices)
                tokens[noun_indice] = random.choice(anto_dct.get(tokens[noun_indice], ['stocking']))
                ct_querys.append(' '.join(tokens))
        counter_sim_sims = self.cal_sim_aug_score(img, querys, ct_querys)
        counter_sims = counter_sim_sims

        tau = 1.2
        minx = min(counter_sims)
        maxx = max(counter_sims)
        norm_similarity = [(counter_sims[ii]-minx)/(maxx-minx)*(tau-1/tau)+1/tau for ii in range(len(counter_sims))]
        caption_sims = torch.tensor(norm_similarity, device=self.device) * caption_sims
        
        
        # _, _, F1 = score(captions, query, lang='en', baseline_path=f'{ckpt_dir}/roberta-large')
        bs = 0
        start_target = 0
        end_target = len(caption_sims)-1
        stack = []
        caption_sims = caption_sims.tolist()
        alpha = 0.001 # prevent from too many segments
        caption_sims = [caption_sims[i]-min(caption_sims)+alpha for i in range(len(caption_sims))]
        
        # score = [0] + caption_sims.tolist() + [0]
        score = [0] + caption_sims + [0]
        # print(score)
        for i in range(len(score)):
            while stack and score[stack[-1]] > score[i]:
                tmp = stack.pop()
                tmp_bs = (i-stack[-1]-1) * score[tmp]
                if tmp_bs > bs:
                    bs = tmp_bs
                    start_target, end_target = stack[-1], i-2
            stack.append(i)
        idxs = list(range(start_target, end_target+1))
        
        counter_sims_score = counter_sims[0].item()
        # counter_sims_score = counter_sims.mean().item()
        return idxs, counter_sims_score

    def html(self,img,idxs,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        start = html_output(idxs[0])
        end = html_output(idxs[-1])
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;)={start}--{end}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,query,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        idxs, ct_score = self.predict(img, query)
        # idxs, ct_score = self.ct_aug_predict(img, query)
        # print(captions)
        # print(query)
        
        prog_step.state[output_var] = idxs
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            html_str = self.html(img, idxs, output_var)
            return idxs, html_str
        return idxs

class MeasureInterpreter():
    step_name = 'MEASURE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        pos_var = parse_result['args']['pos']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return pos_var,output_var

    def html(self,length,output_var):
        # img = html_embed_image(img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        return f"""<div>{output_var}={step_name}={length}</div>"""

    def execute(self,prog_step,inspect=False):
        pos_var,output_var = self.parse(prog_step)
        poses = prog_step.state[pos_var]
        if len(poses) > 0:
            length = len(poses)
        else:
            length = 0
        prog_step.state[output_var] = length
        if inspect:
            # pos_img = prog_step.state[pos_var+'_IMAGE']
            html_str = self.html(length)
            return length, html_str
        return length


class GetInterpreter():
    step_name = 'GET'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        pos_var = parse_result['args']['pos']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,pos_var,output_var

    def html(self,img,out_img,output_var):
        # img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        return f"""<div>{output_var}={step_name}={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,pos_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        poses = prog_step.state[pos_var]
        if len(poses) > 0:
            pos = poses[0]
            out_img = img[pos]
        else:
            box = []
            out_img = img[0]

        prog_step.state[output_var] = out_img
        if inspect:
            # pos_img = prog_step.state[pos_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var)
            return out_img, html_str
        return out_img

class GetAfterInterpreter():
    step_name = 'GET_AFTER'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        pos_var = parse_result['args']['pos']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,pos_var,output_var

    def html(self,img,out_img,output_var):
        # img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        return f"""<div>{output_var}={step_name}={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,pos_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        poses = prog_step.state[pos_var]
        if len(poses) > 0:
            pos = poses[-1]
            pre_pos = list(range(pos, len(img)))
            pos = int((pre_pos[0]+pre_pos[-1])/2)
            out_img = img[pos]
        else:
            poses = []
            out_img = img[0]

        prog_step.state[output_var] = out_img
        if inspect:
            # pos_img = prog_step.state[pos_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var)
            return out_img, html_str
        return out_img

class GetBeforeInterpreter():
    step_name = 'GET_BEFORE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        pos_var = parse_result['args']['pos']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,pos_var,output_var

    def html(self,img,out_img,output_var):
        # img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        return f"""<div>{output_var}={step_name}={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,pos_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        poses = prog_step.state[pos_var]
        if len(poses) > 0:
            pos = poses[0]
            pre_pos = list(range(pos+1))
            pos = int((pre_pos[0]+pre_pos[-1])/2)
            out_img = img[pos]
        else:
            poses = []
            out_img = img[0]

        prog_step.state[output_var] = out_img
        if inspect:
            # pos_img = prog_step.state[pos_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var)
            return out_img, html_str
        return out_img

class GetBetweenInterpreter():
    step_name = 'GET_BETWEEN'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        pos1_var = parse_result['args']['pos1']
        pos2_var = parse_result['args']['pos2']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,pos1_var,pos2_var,output_var

    def html(self,img,out_img,output_var):
        # img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        return f"""<div>{output_var}={step_name}={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,pos1_var,pos2_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        poses1 = prog_step.state[pos1_var]
        poses2 = prog_step.state[pos2_var]
        if len(poses1) > 0 and len(poses2) > 0:
            pos1 = int((poses1[0]+poses1[-1])/2)
            pos2 = int((poses2[0]+poses2[-1])/2)
            if pos1 > pos2: pos1, pos2 = pos2, pos1
            pre_pos = list(range(pos1, pos2+1))
            pos = int((pre_pos[0]+pre_pos[-1])/2)
            out_img = img[pos]
        else:
            poses = []
            out_img = img[0]

        prog_step.state[output_var] = out_img
        if inspect:
            # pos_img = prog_step.state[pos_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var)
            return out_img, html_str
        return out_img



class CaptionInterpreter():
    step_name = 'CAP'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def generate(self,img):
        inputs = self.processor(img,return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    

    def html(self,img,caption,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        caption = html_output(caption)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;)={caption}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        caption = self.predict(img)
        prog_step.state[output_var] = caption
        if inspect:
            html_str = self.html(img, caption, output_var)
            return caption, html_str

        return caption



class TestInterpreter():
    step_name = 'TEST'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        # print(prog_step.state)
        output = prog_step.state[output_var] == prog_step.state["reference"]
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


    
class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def html(self,eval_expression,step_input,step_output,output_var):
        eval_expression = eval_expression.replace('{','').replace('}','')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def execute(self,prog_step,inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['Yes','No']:
                    prog_state[var_name] = var_value.lower()
                # if var_value in ['yes','no']:
                #     prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value
        
        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        # print(step_input)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output





class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        result_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return output_var, result_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)
            
        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var, result_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        prog_step.state[result_var] = output
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        # self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # self.model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.processor = AutoProcessor.from_pretrained(f"{ckpt_dir}/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained(
            f"{ckpt_dir}/blip-vqa-capfilt-large").to(self.device)
        self.model.eval()
        # self.processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # self.model.eval()
        
        
        model_path = f"{ckpt_dir}/llava-v1.5-13b"
        model_name = get_model_name_from_path(model_path)
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = load_pretrained_model(model_path, None, model_name)

        
        
        # self.clip_model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14"
        # )
        
        self.clip_model = CLIPModel.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14"
        )
        
        # load_path = '/home/patrick/codes/templates/lightning-hydra-template/logs/train/runs/2023-08-04_20-47-11/checkpoints/last.ckpt'
        # config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
        # self.clip_model = CLIPModel(config)
        # ckpt = torch.load(load_path, map_location='cpu')
        # state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # new_state_dict = OrderedDict()
        # for k,v in state_dict.items():
        #     if "text_model.embeddings.position_ids" in k: continue
        #     if "vision_model.embeddings.position_ids" in k: continue
        #     if 'model' in k:
        #         k = k[6:]
        #     new_state_dict[k] = v
        # self.clip_model.load_state_dict(new_state_dict)
        # self.clip_model.to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device)  # instructblip-vicuna-7b
        # self.instructblip_model.eval()
        # self.sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.instructblip_processor = AutoProcessor.from_pretrained(f"{ckpt_dir}/instructblip-flan-t5-xl")
        self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
            f"{ckpt_dir}/instructblip-flan-t5-xl").to(self.device)  # instructblip-vicuna-7b
        self.instructblip_model.eval()
        self.sent_model = SentenceTransformer(f'{ckpt_dir}/sentence-transformers_all-MiniLM-L6-v2').to(self.device)
        

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,question,output_var
    
    def calculate_sim(self,inputs):
        img_feats = self.clip_model.get_image_features(inputs['pixel_values'])
        text_feats = self.clip_model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(text_feats,img_feats.t())


    def predict(self,img,question):
        # encoding = self.processor(img,question,return_tensors='pt')
        # encoding = {k:v.to(self.device) for k,v in encoding.items()}
        
        # with torch.no_grad():
        #     outputs = self.model.generate(**encoding)
        
        question += '\nAnswer the question using a single word or phrase.'
        
        cur_prompt = question
        if self.llava_model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        # image = Image.open(img)
        image = img
        image_tensor = self.llava_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                # do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=10,
                # use_cache=True,
            )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.llava_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # dct = json.load(open(os.path.join(Path.home(), 'codes/visjoint/datasets/gqa/vocab.json'))))
        # vocab_tokens = self.processor(text=dct)['input_ids']
        # vocab_tokens = [[30522] + x[1:] for x in vocab_tokens]
        # trie = MarisaTrie(vocab_tokens)
        # with torch.no_grad():
        #     outputs = self.model.generate(**encoding, num_beams=5, 
        #                                   prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()))

        
        # return self.processor.decode(outputs[0], skip_special_tokens=True), 0
        return outputs.lower(), 0
    
    
    def cal_sim_aug_score(self, img, question, responses, querys, ct_querys):
        # querys = ['a photo of ' + query for query in querys]
        # ct_querys = ['a photo of ' + ct_query for ct_query in ct_querys]
        clip_inputs = self.clip_processor(
            text = querys,
            # images=[img]*len(responses),
            images=img,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
            
        ct_clip_inputs = self.clip_processor(
            text=ct_querys,
            # images=[img]*len(ct_responses),
            images=img,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            ct_sims = self.calculate_sim(ct_clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        # counter_sims = sims
                
        return counter_sims
    
    def cal_cap_aug_score(self, img, question, responses, querys, ct_querys):
        
        cap_inputs = self.instructblip_processor(
            text='an image describe ',
            images=img,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            cap_outputs = self.instructblip_model.generate(**cap_inputs)
            cap_outputs = self.instructblip_processor.decode(cap_outputs[0], skip_special_tokens=True)
            cap_embedding = self.sent_model.encode(cap_outputs, convert_to_tensor=True, device=self.device)
            
            responses_embedding = self.sent_model.encode(querys, convert_to_tensor=True, device=self.device)
            ct_responses_embedding = self.sent_model.encode(ct_querys, convert_to_tensor=True, device=self.device)
            
            responses_sims = util.pytorch_cos_sim(responses_embedding, cap_embedding).squeeze(-1)
            ct_responses_sims = util.pytorch_cos_sim(ct_responses_embedding, cap_embedding).squeeze(-1)
            
        counter_sims = responses_sims - ct_responses_sims
        
        return counter_sims
        
        
    def cal_qa_aug_score(self, img, question, responses, querys, ct_querys):
        
        text = [f'Does "{query}" correctly describe the image?' for query in querys]
        ct_text = [f'Does "{query}" correctly describe the image?' for query in ct_querys]
        images = [img] * len(querys)
        
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        ct_inputs = self.processor(
            text=ct_text,
            images=images,
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        # constrained_index = self.instructblip_processor(text=['Yes', 'No', 'yes', 'no'])["input_ids"]
        constrained_index = self.processor(text=['yes', 'no'])["input_ids"] # instruct-blip is case-sensitive model
        constrained_index = [x[1] for x in constrained_index]
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, 
                                            num_beams=1,
                                            return_dict_in_generate=True,
                                            output_scores=True)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            logits = outputs.scores[1]
            contrained_logits = logits[:, constrained_index]
            contrained_logits = torch.softmax(contrained_logits, dim=-1)
            responses_sims =  contrained_logits[:, 0] - contrained_logits[:, 1]
            
            # contrained_logits1 = logits[:, constrained_index[:2]]
            # contrained_logits1 = torch.softmax(contrained_logits1, dim=-1)
            # contrained_logits2 = logits[:, constrained_index[2:]]
            # contrained_logits2 = torch.softmax(contrained_logits2, dim=-1)
            # responses_sims1 =  contrained_logits1[:, 0] - contrained_logits1[:, 1]
            # responses_sims2 =  contrained_logits2[:, 0] - contrained_logits2[:, 1]
            # responses_sims = (responses_sims1 + responses_sims2) / 2
            
            ct_outputs = self.model.generate(**ct_inputs,
                                                num_beams=1,
                                                return_dict_in_generate=True,
                                                output_scores=True)

            ct_logits = ct_outputs.scores[1]
            ct_contrained_logits = ct_logits[:, constrained_index]
            ct_contrained_logits = torch.softmax(ct_contrained_logits, dim=-1)
            ct_responses_sims =  ct_contrained_logits[:, 1] - ct_contrained_logits[:, 0]
                                       
#             ct_contrained_logits1 = ct_logits[:, constrained_index[:2]]
#             ct_contrained_logits1 = torch.softmax(ct_contrained_logits1, dim=-1)
#             ct_contrained_logits2 = logits[:, constrained_index[2:]]
#             ct_contrained_logits2 = torch.softmax(ct_contrained_logits2, dim=-1)
#             ct_responses_sims1 =  ct_contrained_logits1[:, 1] - ct_contrained_logits1[:, 0]
#             ct_responses_sims2 =  ct_contrained_logits2[:, 1] - ct_contrained_logits2[:, 0]
#             ct_responses_sims = (ct_responses_sims1 + ct_responses_sims2) / 2
 
        counter_sims = (responses_sims + ct_responses_sims) / 2
        
        # counter_sims = responses_sims
    
        return counter_sims
            
        
        
    def ct_aug_predict(self, img, question):
        encoding = self.processor(img,question,return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'],
                                          num_beams=3, num_return_sequences=3, output_scores=True, return_dict_in_generate=True)
        self.model.config.vocab_size = self.model.config.text_config.vocab_size
        # transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
        output_length = 1 + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        scores = np.exp(np.sum(transition_scores.cpu().numpy(), axis=1) / output_length)
        responses = self.processor.batch_decode(outputs['sequences'], skip_special_tokens=True)
        
        # # output original results
        # responses, scores = zip(*sorted(zip(responses, scores), key=lambda x:x[1],reverse=True))        
        # return responses[0], 0
        
        # print(responses)
        ############################## AUGMENTED GENERATION ##############################
        
        # querys = []
        # for response in responses:
        #     prompt = f"""Below is an instruction that describes a task.
        #         Turn the Question-Answer pair to a Sentence: 
        #         ### Question: Are there trains or fences in this scene?
        #         ### Answer: no
        #         ### Sentence: There aren't trains or fences in this scene
        #         ### Question: Who is carrying the umbrella?
        #         ### Answer: the man
        #         ### Sentence: The man is carrying the umbrella
        #         ### Question: Is the pillow in the top part or in the bottom of the picture?
        #         ### Answer: top part
        #         ### Sentence: The pillow is in the top part of the picture
        #         ### Question: {question}
        #         ### Answer: {response}
        #         ### Sentence: """
        #     querys.append(prompt)
        # query_inputs = self.instructblip_processor(text=querys, images=[img]*len(querys), return_tensors="pt", padding=True).to(self.device)
        # query_outputs = self.instructblip_model.generate(**query_inputs)
        # querys = self.instructblip_processor.batch_decode(query_outputs, skip_special_tokens=True)
        
        
        querys = [question + response for response in responses]
        
        # get counterfactual responses
        # replace nourn with counter-factual
        ct_querys = []
        for query in querys:
            tag_query = pos_tag(word_tokenize(query))
            tokens = [x[0] for x in tag_query]
            tags = [x[1] for x in tag_query]
            noun_indices = [ii for ii in range(len(tags)) if 'NN' in tags[ii]]
            if noun_indices == []:
                ct_querys.append('not ' + query)
            else:
                noun_indice = random.choice(noun_indices)
                tokens[noun_indice] = random.choice(anto_dct.get(tokens[noun_indice], ['stocking']))
                ct_querys.append(' '.join(tokens))
        
        counter_sim_sims = self.cal_sim_aug_score(img, question, responses, querys, ct_querys)
        # counter_cap_sims = self.cal_cap_aug_score(img, question, responses, querys, ct_querys)
        # counter_qa_sims = self.cal_qa_aug_score(img, question, responses, querys, ct_querys)
        
#         print(counter_sim_sims)
#         print(counter_cap_sims)
#         print(counter_qa_sims)
        
        # way1 weighted sum
        # counter_sims = 0.2* counter_sim_sims + 0.4 * counter_cap_sims + 0.4 * counter_qa_sims
        counter_sims = counter_sim_sims
        # counter_sims = counter_qa_sims
        # # way2 select
        # counter_sim_var = torch.var(counter_sim_sims)
        # counter_cap_var = torch.var(counter_cap_sims)
        # counter_qa_var = torch.var(counter_qa_sims)
        # indice = torch.argmax(torch.tensor([counter_sim_var, counter_cap_var, counter_qa_var]))
        # counter_sims = [counter_sim_sims, counter_cap_sims, counter_qa_sims][indice]
        # # way3 router
        
        # # way1 abstract weight
        tau = 1.2
        minx = min(counter_sims)
        maxx = max(counter_sims)
        norm_similarity = [(counter_sims[ii]-minx)/(maxx-minx)*(tau- 1/tau)+1/tau for ii in range(len(counter_sims))]
        
        # # way2 soft weight
        # tau = 1.2 # scale
        # norm_similarity = torch.softmax(counter_sims, dim=-1)
        # norm_similarity = [norm_similarity[ii] * (tau-1/tau) + 1/tau for ii in range(len(counter_sims))]
        
        # print(norm_similarity)
        # print(scores)

        aug_scores = torch.tensor(norm_similarity) * torch.tensor(scores)
        responses, scores = zip(*sorted(zip(responses, aug_scores), key=lambda x:x[1],reverse=True))
        
        counter_sims_score = counter_sims[0].item()
        # counter_sims_score = counter_sims.mean().item()
        return responses[0], counter_sims_score
    
        ##########################################################################################
        
    
    def aug_predict(self,img,question):
        
        # generate top k results
        encoding = self.processor(img,question,return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'],
                                          num_beams=5, num_return_sequences=5, output_scores=True, return_dict_in_generate=True)
        self.model.config.vocab_size = self.model.config.text_config.vocab_size
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
        output_length = 1 + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        scores = np.exp(np.sum(transition_scores.cpu().numpy(), axis=1) / output_length)
        responses = self.processor.batch_decode(outputs['sequences'], skip_special_tokens=True)
        
        openai_cache_file = 'openai_cache.pkl'
        if not os.path.exists(openai_cache_file):
            with open(openai_cache_file, 'wb') as jp:
                pickle.dump({}, jp)
        with open(openai_cache_file) as jp:
            openai_cache = json.load(jp)
        
        
        # calculate the similarity
        
        querys = []
        for response in responses:
            prompt = f"""Below is an instruction that describes a task.
                Turn the Question-Answer pair to a Sentence: 
                ### Question: Are there trains or fences in this scene?
                ### Answer: no
                ### Sentence: There aren't trains or fences in this scene
                ### Question: Who is carrying the umbrella?
                ### Answer: the man
                ### Sentence: The man is carrying the umbrella
                ### Question: Is the pillow in the top part or in the bottom of the picture?
                ### Answer: top part
                ### Sentence: The pillow is in the top part of the picture
                ### Question: Which side is the food on?
                ### Answer: RIGHT
                ### Sentence: The food is on right
                ### Question: What do the wetsuit and the sky have in common?
                ### Answer: they are blue
                ### Sentence: Both the wetsuit and the sky are blue
                ### Question: {question}
                ### Answer: {response}
                ### Sentence: """
            if prompt not in openai_cache:
                messages = [
                    {'role': 'user', 'content': prompt}
                ]
                query = openai_api_stream(messages).strip()
                querys.append(query)
                openai_cache[prompt] = querys
            else:
                querys = openai_cache[prompt]
        
        clip_inputs = self.clip_processor(
            text=querys, images=[img], return_tensors='pt', padding=True
        ).to(self.device)
        with torch.no_grad():
            similarity = self.calculate_sim(clip_inputs).squeeze(0).cpu().numpy()
        
        # print(similarity)
        # print(scores)

        minx = min(similarity)
        maxx = max(similarity)
        tau = 1.0 # 0-2
        norm_similarity = [(similarity[ii]-minx)/(maxx-minx)*tau+(1-tau/2) for ii in range(len(similarity))]
        aug_scores = torch.tensor(norm_similarity) * torch.tensor(scores)
        responses, scores = zip(*sorted(zip(responses, aug_scores), key=lambda x:x[1],reverse=True))
        
        # joint_scores = [similarity[ii] + scores[ii] for ii in range(len(similarity))]
        # responses, scores = zip(*sorted(zip(responses, joint_scores), key=lambda x:x[1],reverse=True))

        
        # print(responses)
        # print(scores)
        return responses[0]
    

    def html(self,img,question,answer,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        answer = html_output(answer)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        question_arg = html_arg_name('question')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;{question_arg}='{question}')={answer}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,question,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        answer, ct_score = self.predict(img,question)
        # answer = self.aug_predict(img,question)
        # answer, ct_score = self.ct_aug_predict(img, question)
        prog_step.state[output_var] = answer
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            html_str = self.html(img, question, answer, output_var)
            return answer, html_str

        return answer


class LocInterpreter():
    step_name = 'LOC'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        # self.processor = OwlViTProcessor.from_pretrained(
        #     "google/owlvit-large-patch14")
        # self.model = OwlViTForObjectDetection.from_pretrained(
        #     "google/owlvit-large-patch14").to(self.device)
        self.processor = OwlViTProcessor.from_pretrained(
            f"{ckpt_dir}/owlvit-large-patch14")
        self.model = OwlViTForObjectDetection.from_pretrained(
            f"{ckpt_dir}/owlvit-large-patch14").to(self.device)
        self.model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh
        
        # for evaluation
        self.clip_model = CLIPModel.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14")
        
        # load_path = '/home/patrick/codes/sft/logs/gqa/train/runs/clip_gqa/checkpoints/last.ckpt'
        # config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
        # self.clip_model = CLIPModel(config)
        # ckpt = torch.load(load_path, map_location='cpu')
        # state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # new_state_dict = OrderedDict()
        # for k,v in state_dict.items():
        #     if 'text_model.embeddings.position_ids' in k or 'vision_model.embeddings.position_ids' in k:
        #         continue
        #     if 'model' in k:
        #         k = k[6:]
        #     new_state_dict[k] = v
        # self.clip_model.load_state_dict(new_state_dict)
        # self.clip_model.to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # self.instructblip_model.eval()
        
        # self.sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        
        # self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # self.blip_model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-capfilt-large").to(self.device)
        # self.blip_model.eval()
        
        
        # self.joint_module = nn.Sequential(
        #     nn.Linear(self.clip_model.projection_dim, self.clip_model.projection_dim),
        #     nn.Tanh()
        # )

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']], 
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return [], 0

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        return selected_boxes, 0

    def top_box(self,img):
        w,h = img.size        
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1

    def html(self,img,box_img,output_var,obj_name):
        step_name=html_step_name(self.step_name)
        obj_arg=html_arg_name('object')
        img_arg=html_arg_name('image')
        output_var=html_var_name(output_var)
        img=html_embed_image(img)
        box_img=html_embed_image(box_img,300)
        return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"


    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        ct_score = 0
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            # bboxes, ct_score = self.predict(img,obj_name)
            
            # # deprecated singular norm will cause some errors like buses -> bu
            # sing_obj_name = inflect_engine.singular_noun(obj_name)
            # if sing_obj_name:
            #     obj_name = sing_obj_name

            bboxes, ct_score = self.ct_aug_predict(img,obj_name)
            
            # bboxes, ct_score = self.aug_predict(img,obj_name)
            # bboxes = self.aug_cap_predict(img,obj_name)
            # bboxes = self.aug_qa_predict(img,obj_name)

        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var+'_IMAGE'] = box_img
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return bboxes
    
    def calculate_sim(self,inputs):
        img_feats = self.clip_model.get_image_features(inputs['pixel_values'])
        text_feats = self.clip_model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())
    
    def calculate_text_sim(self,inputs1, inputs2):
        text1_feats = self.clip_model.get_text_features(inputs1['input_ids'])
        text2_feats = self.clip_model.get_text_features(inputs2['input_ids'])
        text12_feats = text1_feats / text1_feats.norm(p=2, dim=-1, keepdim=True)
        text21_feats = text2_feats / text2_feats.norm(p=2, dim=-1, keepdim=True)
        return (torch.matmul(text12_feats,text21_feats.t()) + torch.matmul(text21_feats,text12_feats.t())) / 2
    
    def cal_sim_aug_score(self, img, boxes, obj_name):
        text = f'a photo of {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'a photo of {ct_obj}'
        
        # print(text, ct_text)
        
        objs = [img.crop(o) for o in boxes]
        clip_inputs = self.clip_processor(
            text=text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        clip_inputs = self.clip_processor(
            text=ct_text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # ct_sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()    
            ct_sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        return counter_sims
    
    def cal_cap_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'an image describe: {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'an image describe: {ct_obj}'
        
        objs = [img.crop(o) for o in boxes]
        texts = ['an image describe: '] * len(objs)
        # instructblip_inputs = self.instructblip_processor(
        #     text=texts, images=objs, return_tensors='pt'
        # ).to(self.device)
        blip_inputs = self.blip_processor(objs, texts, return_tensors='pt').to(self.device)
        text_embedding = self.sent_model.encode(text, convert_to_tensor=True, device=self.device)
        counter_text_embedding = self.sent_model.encode(ct_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            outputs = self.blip_model.generate(**blip_inputs)
            responses = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
            # outputs = self.instructblip_model.generate(**instructblip_inputs)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            # sims = bert_score(responses, [text])
            responses = ['an image describe: '+x for x in responses]
            responses_embedding = self.sent_model.encode(responses, convert_to_tensor=True, device=self.device)
            # responses_embedding_norm = torch.nn.functional.normalize(responses_embedding, p=2, dim=1)
            # b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
            # return torch.mm(a_norm, b_norm.transpose(0, 1))
            responses_sims = util.pytorch_cos_sim(responses_embedding, text_embedding).squeeze(-1)
            ct_responses_sims = util.pytorch_cos_sim(responses_embedding, counter_text_embedding).squeeze(-1)
            
        # print('response', responses_sims)
        # print('counter response', ct_responses_sims)
        counter_sims = responses_sims - ct_responses_sims
        
        return counter_sims
    
    def cal_qa_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'are there {obj_name} in the image?'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'are there {ct_obj} in the image'
        
        objs = [img.crop(o) for o in boxes]
        texts = [text] * len(objs)
        ct_texts = [ct_text] * len(objs)
        blip_inputs = self.blip_processor(objs, texts, return_tensors="pt").to(self.device)
        ct_blip_inputs = self.blip_processor(objs, ct_texts, return_tensors="pt").to(self.device)
        # instructblip_inputs = self.instructblip_processor(
        #     text=texts, images=objs, return_tensors='pt'
        # ).to(self.device)
        # ct_instructblip_inputs = self.instructblip_processor(
        #     text=ct_texts, images=objs, return_tensors='pt'
        # ).to(self.device)
        # constrained_index = self.instructblip_processor(text=['Yes', 'No', 'yes', 'no'])["input_ids"]
        # constrained_index = self.instructblip_processor(text=['Yes', 'No'])["input_ids"]
        
        # constrained_index = self.instructblip_processor(text=['yes', 'no'])["input_ids"]
        # constrained_index = [x[0] for x in constrained_index]
        
        constrained_index = self.blip_processor(text=['yes', 'no'])["input_ids"]
        constrained_index = [x[1] for x in constrained_index]
        with torch.no_grad():
            # outputs = self.instructblip_model.generate(**instructblip_inputs, 
            #                                             num_beams=1,
            #                                             return_dict_in_generate=True,
            #                                             output_scores=True)
            outputs = self.blip_model.generate(**blip_inputs, 
                                                        num_beams=1,
                                                        return_dict_in_generate=True,
                                                        output_scores=True)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            logits = outputs.scores[1]
            contrained_logits1 = logits[:, constrained_index[:2]]
            contrained_logits1 = torch.softmax(contrained_logits1, dim=-1)
            # contrained_logits2 = logits[:, constrained_index[2:]]
            # contrained_logits2 = torch.softmax(contrained_logits2, dim=-1)
            responses_sims1 =  contrained_logits1[:, 0] - contrained_logits1[:, 1]
            # responses_sims2 =  contrained_logits2[:, 0] - contrained_logits2[:, 1]
            # responses_sims = (responses_sims1 + responses_sims2) / 2
            responses_sims = responses_sims1
            # ct_outputs = self.instructblip_model.generate(**ct_instructblip_inputs,
            #                                                 num_beams=1,
            #                                                 return_dict_in_generate=True,
            #                                                 output_scores=True)
            ct_outputs = self.blip_model.generate(**ct_blip_inputs,
                                                    num_beams=1,
                                                    return_dict_in_generate=True,
                                                    output_scores=True)
            # ct_responses = self.instructblip_processor.batch_decode(ct_outputs, skip_special_tokens=True)
            ct_logits = ct_outputs.scores[1]
            ct_contrained_logits1 = ct_logits[:, constrained_index[:2]]
            ct_contrained_logits1 = torch.softmax(ct_contrained_logits1, dim=-1)
            # ct_contrained_logits2 = logits[:, constrained_index[2:]]
            # ct_contrained_logits2 = torch.softmax(ct_contrained_logits2, dim=-1)
            ct_responses_sims1 =  ct_contrained_logits1[:, 1] - ct_contrained_logits1[:, 0]
            # ct_responses_sims2 =  ct_contrained_logits2[:, 1] - ct_contrained_logits2[:, 0]
            # ct_responses_sims = (ct_responses_sims1 + ct_responses_sims2) / 2
            ct_responses_sims = ct_responses_sims1
 
        counter_sims = (responses_sims + ct_responses_sims) / 2
        
        return counter_sims
    
    
    def ct_aug_predict(self, img, obj_name):
        
        # self.thresh = 0.08
        
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']], 
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return [], 0

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
    
        counter_sim_sims = self.cal_sim_aug_score(img, selected_boxes, obj_name)
        # counter_cap_sims = self.cal_cap_aug_score(img, selected_boxes, obj_name)
        # counter_qa_sims = self.cal_qa_aug_score(img, selected_boxes, obj_name)
        
#         print(counter_sim_sims)
#         print(counter_cap_sims)
#         print(counter_qa_sims)
        
        # # way1 weighted sum
        # counter_sims = 0.4 * counter_sim_sims + 0.3 * counter_cap_sims + 0.3 * counter_qa_sims
        counter_sims = counter_sim_sims
        # counter_sims = counter_qa_sims
        # # way2 select 
        # counter_sim_var = torch.var(counter_sim_sims)
        # counter_cap_var = torch.var(counter_cap_sims)
        # counter_qa_var = torch.var(counter_qa_sims)
        # indice = torch.argmax(torch.tensor([counter_sim_var, counter_cap_var, counter_qa_var]))
        # counter_sims = [counter_sim_sims, counter_cap_sims, counter_qa_sims][indice]
        # # way3 router
        
        minx = min(counter_sims)
        maxx = max(counter_sims)
        # tau = 1.0 
        # norm_counter_scores = [(counter_sims[ii]-minx)/(maxx-minx) * tau + (1 - tau/2) for ii in range(len(counter_sims))]
        tau = 2
        # norm_counter_scores = torch.softmax(counter_sims, dim=-1)
        # norm_counter_scores = [norm_counter_scores[ii] * (tau-1/tau) + 1/tau for ii in range(len(norm_counter_scores))]
        norm_counter_scores = [(counter_sims[ii]-minx)/(maxx-minx) * (tau - 1/tau) + 1/tau for ii in range(len(counter_sims))]
        aug_sims = torch.tensor(norm_counter_scores) * torch.tensor(selected_scores)
        aug_sims = aug_sims.tolist()
        boxes, scores = zip(*sorted(zip(selected_boxes, aug_sims), key=lambda x:x[1], reverse=True))
        
        selected_boxes = []
        selected_scores = []
        
        # self.thresh = 0.1
        
        for i in range(len(scores)):
            
            # if counter_scores[i] >= 0.05: # counter factual scores
            if scores[i] > self.thresh: # counter factual scores
                selected_boxes.append(boxes[i])
                selected_scores.append(scores[i])
        
        counter_sims_score = counter_sims.mean().item()
        
        return selected_boxes, counter_sims_score
    

class Loc2Interpreter(LocInterpreter):

    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        # bboxes, ct_score = self.predict(img,obj_name)
        bboxes, ct_score = self.ct_aug_predict(img,obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))
        prog_step.state[output_var] = objs
        prog_step.state['CT_SCORE'] += ct_score

        if inspect:
            box_img = self.box_image(img, bboxes, highlight_best=False)
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return objs


class CountInterpreter():
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return box_var,output_var

    def html(self,box_img,output_var,count):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_img = html_embed_image(box_img)
        output = html_output(count)
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        box_var,output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count = len(boxes)
        prog_step.state[output_var] = count
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(box_img, output_var, count)
            return count, html_str

        return count


class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,box_var,output_var

    def html(self,img,out_img,output_var,box_img):
        img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            box = []
            out_img = img

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [cx,0,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            right_box = self.right_of(box, img.size)
        else:
            w,h = img.size
            box = []
            right_box = [int(w/2),0,w-1,h-1]
        
        out_img = img.crop(right_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [0,0,cx,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            left_box = self.left_of(box, img.size)
        else:
            w,h = img.size
            box = []
            left_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(left_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,0,w-1,cy]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            above_box = self.above(box, img.size)
        else:
            w,h = img.size
            box = []
            above_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(above_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,cy,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            below_box = self.below(box, img.size)
        else:
            w,h = img.size
            box = []
            below_box = [0,0,int(w/2),h-1]
        
        out_img = img.crop(below_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'


class SegmentInterpreter():
    step_name = 'SEG'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        # self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
        #     "facebook/maskformer-swin-base-coco")
        # self.model = MaskFormerForInstanceSegmentation.from_pretrained(
        #     "facebook/maskformer-swin-base-coco").to(self.device)
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            f"{ckpt_dir}/maskformer-swin-base-coco")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            f"{ckpt_dir}/maskformer-swin-base-coco").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def pred_seg(self,img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
        instance_map = outputs['segmentation'].cpu().numpy()
        objs = []
        print(outputs.keys())
        for seg in outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            mask = (instance_map==inst_id).astype(float)
            resized_mask = np.array(
                Image.fromarray(mask).resize(
                    img.size,resample=Image.BILINEAR))
            Y,X = np.where(resized_mask>0.5)
            x1,x2 = np.min(X), np.max(X)
            y1,y2 = np.min(Y), np.max(Y)
            num_pixels = np.sum(mask)
            objs.append(dict(
                mask=resized_mask,
                category=category,
                box=[x1,y1,x2,y2],
                inst_id=inst_id
            ))

        return objs

    def html(self,img_var,output_var,output):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        img_arg = html_arg_name('image')
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.pred_seg(img)
        prog_step.state[output_var] = objs
        if inspect:
            labels = [str(obj['inst_id'])+':'+obj['category'] for obj in objs]
            obj_img = vis_masks(img, objs, labels)
            html_str = self.html(img_var, output_var, obj_img)
            return objs, html_str

        return objs


class SelectInterpreter():
    step_name = 'SELECT'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        # self.model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.model.eval()
        # self.processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14")
        
        # # for verification
        # self.clip_model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        # self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # self.blip_model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.blip_processor = AutoProcessor.from_pretrained(f"{ckpt_dir}/blip-vqa-capfilt-large")
        self.blip_model = BlipForQuestionAnswering.from_pretrained(
            f"{ckpt_dir}/blip-vqa-capfilt-large").to(self.device)
        self.blip_model.eval()
        # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # self.instructblip_model.eval()
        # self.sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.sent_model = SentenceTransformer(f'{ckpt_dir}/sentence-transformers_all-MiniLM-L6-v2').to(self.device)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = eval(parse_result['args']['query']).split(',')
        category = eval(parse_result['args']['category'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query,category,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu().numpy()
            
        obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids]
    
    def ct_aug_query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            # scores = self.calculate_sim(inputs).cpu().numpy() # num_obj * num_query
            scores = self.calculate_sim(inputs).cpu()
        
        score_weight = torch.ones(scores.shape)
        
        boxs = [obj['box'] for obj in objs]
        
        ct_score = 0
        
        for idx, q in enumerate(query):
            counter_sim_sims = self.cal_sim_aug_score(img, boxs, q)
            counter_cap_sims = self.cal_cap_aug_score(img, boxs, q)
            counter_qa_sims = self.cal_qa_aug_score(img, boxs, q)
            
            counter_sims = 0.4 * counter_sim_sims + 0.3 * counter_cap_sims + 0.3 * counter_qa_sims
            
            tau = 1.5
            norm_counter_scores = torch.softmax(counter_sims, dim=-1)
            norm_counter_scores = [norm_counter_scores[ii] * (tau-1/tau) + 1/tau for ii in range(len(norm_counter_scores))]
            score_weight[:,idx] = torch.tensor(norm_counter_scores)
            # aug_sims = torch.tensor(norm_counter_scores) * torch.tensor(selected_scores)
            ct_score += counter_sims.mean().item()
            
        if len(query): ct_score /= len(query)
            
        scores = scores * score_weight
            
        obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids], ct_score

    def html(self,img_var,obj_var,query,category,output_var,output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        category_arg = html_arg_name('category')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{query_arg}={query},{category_arg}={category})={output}</div>"""

    def query_string_match(self,objs,q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q,f'{q}-merged',f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category']==cat]
        
        return None

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,query,category,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        select_objs = []
        ct_score = 0

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs


        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue
                
                select_objs += matches

        if query is not None and len(select_objs)==0:
            # select_objs = self.query_obj(query, objs, img)
            select_objs, ct_score = self.ct_aug_query_obj(query, objs, img)

        prog_step.state[output_var] = select_objs
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            select_obj_img = vis_masks(img, select_objs)
            html_str = self.html(img_var, obj_var, query, category, output_var, select_obj_img)
            return select_objs, html_str
        

        return select_objs
    
    def cal_sim_aug_score(self, img, boxes, obj_name):
        text = f'a photo of {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'a photo of {ct_obj}'
        
        # print(text, ct_text)
        
        objs = [img.crop(o) for o in boxes]
        clip_inputs = self.processor(
            text=text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        clip_inputs = self.processor(
            text=ct_text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # ct_sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()    
            ct_sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        return counter_sims
    
    def cal_cap_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'an image describe {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'an image describe {ct_obj}'
        
        objs = [img.crop(o) for o in boxes]
        texts = ['an image describe '] * len(objs)
        instructblip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        text_embedding = self.sent_model.encode(text, convert_to_tensor=True, device=self.device)
        counter_text_embedding = self.sent_model.encode(ct_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            outputs = self.blip_model.generate(**instructblip_inputs)
            responses = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
            # sims = bert_score(responses, [text])
            responses = ['an image describe '+x for x in responses]
            responses_embedding = self.sent_model.encode(responses, convert_to_tensor=True, device=self.device)
            # responses_embedding_norm = torch.nn.functional.normalize(responses_embedding, p=2, dim=1)
            # b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
            # return torch.mm(a_norm, b_norm.transpose(0, 1))
            responses_sims = util.pytorch_cos_sim(responses_embedding, text_embedding).squeeze(-1)
            ct_responses_sims = util.pytorch_cos_sim(responses_embedding, counter_text_embedding).squeeze(-1)
            
        # print('response', responses_sims)
        # print('counter response', ct_responses_sims)
        counter_sims = responses_sims - ct_responses_sims
        
        return counter_sims
    
    def cal_qa_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'are there {obj_name} in the image?'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'are there {ct_obj} in the image'
        
        objs = [img.crop(o) for o in boxes]
        texts = [text] * len(objs)
        ct_texts = [ct_text] * len(objs)
        blip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        ct_blip_inputs = self.blip_processor(
            text=ct_texts, images=objs, return_tensors='pt'
        ).to(self.device)
        # constrained_index = self.blip_processor(text=['Yes', 'No', 'yes', 'no'])["input_ids"]
        # constrained_index = [x[0] for x in constrained_index]
        constrained_index = self.blip_processor(text=['yes', 'no'])["input_ids"]
        constrained_index = [x[1] for x in constrained_index]
        with torch.no_grad():
            outputs = self.blip_model.generate(**blip_inputs, 
                                                        num_beams=1,
                                                        return_dict_in_generate=True,
                                                        output_scores=True)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            logits = outputs.scores[1]
            contrained_logits1 = logits[:, constrained_index[:2]]
            contrained_logits1 = torch.softmax(contrained_logits1, dim=-1)
            # contrained_logits2 = logits[:, constrained_index[2:]]
            # contrained_logits2 = torch.softmax(contrained_logits2, dim=-1)
            responses_sims1 =  contrained_logits1[:, 0] - contrained_logits1[:, 1]
            # responses_sims2 =  contrained_logits2[:, 0] - contrained_logits2[:, 1]
            # responses_sims = (responses_sims1 + responses_sims2) / 2
            responses_sims = responses_sims1
            ct_outputs = self.blip_model.generate(**ct_blip_inputs,
                                                            num_beams=1,
                                                            return_dict_in_generate=True,
                                                            output_scores=True)
            # ct_responses = self.instructblip_processor.batch_decode(ct_outputs, skip_special_tokens=True)
            ct_logits = ct_outputs.scores[1]
            ct_contrained_logits1 = ct_logits[:, constrained_index[:2]]
            ct_contrained_logits1 = torch.softmax(ct_contrained_logits1, dim=-1)
            # ct_contrained_logits2 = logits[:, constrained_index[2:]]
            # ct_contrained_logits2 = torch.softmax(ct_contrained_logits2, dim=-1)
            ct_responses_sims1 =  ct_contrained_logits1[:, 1] - ct_contrained_logits1[:, 0]
            # ct_responses_sims2 =  ct_contrained_logits2[:, 1] - ct_contrained_logits2[:, 0]
            # ct_responses_sims = (ct_responses_sims1 + ct_responses_sims2) / 2
            ct_responses_sims = ct_responses_sims1
 
        counter_sims = (responses_sims + ct_responses_sims) / 2
        
        return counter_sims
    
    


class ColorpopInterpreter():
    step_name = 'COLORPOP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        gimg = img.copy()
        gimg = gimg.convert('L').convert('RGB')
        gimg = np.array(gimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            gimg = mask*img + (1-mask)*gimg

        gimg = np.array(gimg).astype(np.uint8)
        gimg = Image.fromarray(gimg)
        prog_step.state[output_var] = gimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, gimg)
            return gimg, html_str

        return gimg


class BgBlurInterpreter():
    step_name = 'BGBLUR'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def smoothen_mask(self,mask):
        mask = Image.fromarray(255*mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius = 5))
        return np.array(mask).astype(float)/255

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        bgimg = img.copy()
        bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius = 2))
        bgimg = np.array(bgimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            mask = self.smoothen_mask(mask)
            bgimg = mask*img + (1-mask)*bgimg

        bgimg = np.array(bgimg).astype(np.uint8)
        bgimg = Image.fromarray(bgimg)
        prog_step.state[output_var] = bgimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, bgimg)
            return bgimg, html_str

        return bgimg


class FaceDetInterpreter():
    step_name = 'FACEDET'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.model = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def box_image(self,img,boxes):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            draw.rectangle(box,outline='blue',width=5)

        return img1

    def enlarge_face(self,box,W,H,f=1.5):
        x1,y1,x2,y2 = box
        w = int((f-1)*(x2-x1)/2)
        h = int((f-1)*(y2-y1)/2)
        x1 = max(0,x1-w)
        y1 = max(0,y1-h)
        x2 = min(W,x2+w)
        y2 = min(H,y2+h)
        return [x1,y1,x2,y2]

    def det_face(self,img):
        with torch.no_grad():
            faces = self.model.detect(np.array(img))
        
        W,H = img.size
        objs = []
        for i,box in enumerate(faces):
            x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
            x1,y1,x2,y2 = self.enlarge_face([x1,y1,x2,y2],W,H)
            mask = np.zeros([H,W]).astype(float)
            mask[y1:y2,x1:x2] = 1.0
            objs.append(dict(
                box=[x1,y1,x2,y2],
                category='face',
                inst_id=i,
                mask = mask
            ))
        return objs

    def html(self,img,output_var,objs):
        step_name = html_step_name(self.step_name)
        box_img = self.box_image(img, [obj['box'] for obj in objs])
        img = html_embed_image(img)
        box_img = html_embed_image(box_img,300)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({img_arg}={img})={box_img}</div>"""


    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.det_face(img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(img, output_var, objs)
            return objs, html_str

        return objs


class EmojiInterpreter():
    step_name = 'EMOJI'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        emoji_name = eval(parse_result['args']['emoji'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,emoji_name,output_var

    def add_emoji(self,objs,emoji_name,img):
        W,H = img.size
        emojipth = os.path.join(EMOJI_DIR,f'smileys/{emoji_name}.png')
        for obj in objs:
            x1,y1,x2,y2 = obj['box']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            s = (y2-y1)/1.5
            x_pos = (cx-0.5*s)/W
            y_pos = (cy-0.5*s)/H
            emoji_size = s/H
            emoji_aug = imaugs.OverlayEmoji(
                emoji_path=emojipth,
                emoji_size=emoji_size,
                x_pos=x_pos,
                y_pos=y_pos)
            img = emoji_aug(img)

        return img

    def html(self,img_var,obj_var,emoji_name,output_var,img):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        emoji_arg = html_arg_name('emoji')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img = html_embed_image(img,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{emoji_arg}='{emoji_name}')={img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,emoji_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.add_emoji(objs, emoji_name, img)
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, obj_var, emoji_name, output_var, img)
            return img, html_str

        return img


class ListInterpreter():
    step_name = 'LIST'

    prompt_template = """
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:"""

    def __init__(self):
        print(f'Registering {self.step_name} step')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        text = eval(parse_result['args']['query'])
        list_max = eval(parse_result['args']['max'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return text,list_max,output_var

    def get_list(self,text,list_max):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.prompt_template.format(list_max=list_max,text=text),
            temperature=0.7,
            max_tokens=256,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )

        item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
        return item_list

    def html(self,text,list_max,item_list,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        max_arg = html_arg_name('max')
        output = html_output(item_list)
        return f"""<div>{output_var}={step_name}({query_arg}='{text}', {max_arg}={list_max})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        text,list_max,output_var = self.parse(prog_step)
        item_list = self.get_list(text,list_max)
        prog_step.state[output_var] = item_list
        if inspect:
            html_str = self.html(text, list_max, item_list, output_var)
            return item_list, html_str

        return item_list


class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        image_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        category_var = parse_result['args']['categories']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return image_var,obj_var,category_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        if len(objs)==0:
            images = [img]
            return []
        else:
            images = [img.crop(obj['box']) for obj in objs]

        if len(query)==1:
            query = query + ['other']

        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            sim = self.calculate_sim(inputs)
            

        # if only one query then select the object with the highest score
        if len(query)==1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class']=query[0]
            obj['class_score'] = 100.0*scores[obj_ids[0],0]
            return [obj]

        # assign the highest scoring class to each object but this may assign same class to multiple objects
        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)
        for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i,cat_id]
            obj['class'] = class_name #+ f'({score_str})'
            obj['class_score'] = round(class_score*100,1)

        # sort by class scores and then for each class take the highest scoring object
        objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
        objs = [obj for obj in objs if 'class' in obj]
        classes = set([obj['class'] for obj in objs])
        new_objs = []
        for class_name in classes:
            cls_objs = [obj for obj in objs if obj['class']==class_name]

            max_score = 0
            max_obj = None
            for obj in cls_objs:
                if obj['class_score'] > max_score:
                    max_obj = obj
                    max_score = obj['class_score']

            new_objs.append(max_obj)

        return new_objs

    def html(self,img_var,obj_var,objs,cat_var,output_var):
        step_name = html_step_name(self.step_name)
        output = []
        for obj in objs:
            output.append(dict(
                box=obj['box'],
                tag=obj['class'],
                score=obj['class_score']
            ))
        output = html_output(output)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        cat_var = html_var_name(cat_var)
        obj_var = html_var_name(obj_var)
        img_arg = html_arg_name('image')
        cat_arg = html_arg_name('categories')
        return f"""<div>{output_var}={step_name}({img_arg}={img_var},{cat_arg}={cat_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        image_var,obj_var,category_var,output_var = self.parse(prog_step)
        img = prog_step.state[image_var]
        objs = prog_step.state[obj_var]
        cats = prog_step.state[category_var]
        objs = self.query_obj(cats, objs, img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(image_var,obj_var,objs,category_var,output_var)
            return objs, html_str

        return objs


class TagInterpreter():
    step_name = 'TAG'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def tag_image(self,img,objs):
        W,H = img.size
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
        for i,obj in enumerate(objs):
            box = obj['box']
            draw.rectangle(box,outline='green',width=4)
            x1,y1,x2,y2 = box
            if 'class' in obj:
                label = obj['class'] + '({})'.format(obj['class_score'])
            else:
                label = 'NA'
            if 'class' in obj:
                w,h = font.getsize(label)
                if x1+w > W or y2+h > H:
                    draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
                    draw.text((x1,y2-h),label,fill='white',font=font)
                else:
                    draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
                    draw.text((x1,y2),label,fill='white',font=font)
        return img1

    def html(self,img_var,tagged_img,obj_var,output_var):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        tagged_img = html_embed_image(tagged_img,300)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('objects')
        output_var = html_var_name(output_var)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var}, {obj_arg}={obj_var})={tagged_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        original_img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.tag_image(original_img, objs)
        prog_step.state['BOX'] = [obj['box'] for obj in objs]
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, img, obj_var, output_var)
            return img, html_str

        return img


def dummy(images, **kwargs):
    return images, [False]*len(images)

class ReplaceInterpreter():
    step_name = 'REPLACE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        device = "cuda"
        model_name = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = dummy

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        prompt = eval(parse_result['args']['prompt'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,prompt,output_var

    def create_mask_img(self,objs):
        mask = objs[0]['mask']
        mask[mask>0.5] = 255
        mask[mask<=0.5] = 0
        mask = mask.astype(np.uint8)
        return Image.fromarray(mask)

    def merge_images(self,old_img,new_img,mask):
        print(mask.size,old_img.size,new_img.size)

        mask = np.array(mask).astype(np.float)/255
        mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
        img = mask*np.array(new_img) + (1-mask)*np.array(old_img)
        return Image.fromarray(img.astype(np.uint8))

    def resize_and_pad(self,img,size=(512,512)):
        new_img = Image.new(img.mode,size)
        thumbnail = img.copy()
        thumbnail.thumbnail(size)
        new_img.paste(thumbnail,(0,0))
        W,H = thumbnail.size
        return new_img, W, H

    def predict(self,img,mask,prompt):
        mask,_,_ = self.resize_and_pad(mask)
        init_img,W,H = self.resize_and_pad(img)
        new_img = self.pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            # strength=0.98,
            guidance_scale=7.5,
            num_inference_steps=50 #200
        ).images[0]
        return new_img.crop((0,0,W-1,H-1)).resize(img.size)

    def html(self,img_var,obj_var,prompt,output_var,output):
        step_name = html_step_name(img_var)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        prompt_arg = html_arg_name('prompt')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var},{prompt_arg}='{prompt}')={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,prompt,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        mask = self.create_mask_img(objs)
        new_img = self.predict(img, mask, prompt)
        prog_step.state[output_var] = new_img
        if inspect:
            html_str = self.html(img_var, obj_var, prompt, output_var, new_img)
            return new_img, html_str
        return new_img

class SegmentsInterpreter():
    step_name = 'SEGS'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            "facebook/maskformer-swin-base-coco")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def pred_seg(self,imgs):
        objss = []
        for img in imgs:
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            label_ids_to_fuse = set()
            outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs, label_ids_to_fuse=label_ids_to_fuse)[0]
            instance_map = outputs['segmentation'].cpu().numpy()
            objs = []
            # print(outputs.keys())
            for seg in outputs['segments_info']:
                inst_id = seg['id']
                label_id = seg['label_id']
                category = self.model.config.id2label[label_id]
                mask = (instance_map==inst_id).astype(float)
                resized_mask = np.array(
                    Image.fromarray(mask).resize(
                        img.size,resample=Image.BILINEAR))
                Y,X = np.where(resized_mask>0.5)
                x1,x2 = np.min(X), np.max(X)
                y1,y2 = np.min(Y), np.max(Y)
                num_pixels = np.sum(mask)
                objs.append(dict(
                    mask=resized_mask,
                    category=category,
                    box=[x1,y1,x2,y2],
                    inst_id=inst_id
                ))
            objss.append(objs)

        return objss

    def html(self,img_var,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        img_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({img_arg}={img_var})</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.pred_seg(img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(img_var, output_var)
            return objs, html_str

        return objs

class AlignInterpreter():
    step_name = 'ALIGN'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        
        #         # for verification
        # self.clip_model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.blip_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.blip_model.eval()
        # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # self.instructblip_model.eval()
        self.sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        part_var = parse_result['args']['part']
        # query_var = parse_result['args']['part']
        query = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,part_var,query,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,parts,objs,img):
        
        # parts score
        if objs == []: 
            # print('no segments')
            # images = [img]
            return 0
        else:
            images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in parts]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu().numpy() # num_objs, num_parts
        
        # obj_ids = scores.argmax(0)    
        # scores = [scores[i,idx] for idx, i in enumerate(obj_ids)]
        
        obj_ids = scores.argmax(1)    
        scores = [scores[i,idx] for i, idx in enumerate(obj_ids)] # num_objs
        # parts = set([parts[idx] for idx in enumerate(obj_ids)])
        
        return sum(scores)/len(scores)
        
        # return sum(scores)/len(scores)
    
    def ct_aug_query_obj(self,parts,objs,img,obj_name):
        if objs == []:
            return 0, 0, ''
        else:
            images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q} of {obj_name}' for q in parts]
        inputs = self.processor(
            text=text, images=images, return_tensors='pt', padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu()
            
        score_weight = torch.ones(scores.shape)
        
        ct_score = 0
        
        boxs = [obj['box'] for obj in objs]
        for idx, q in enumerate(parts):
            # counter_sim_sims = self.cal_sim_aug_score(img, boxs, q)
            # counter_cap_sims = self.cal_cap_aug_score(img, boxs, q)
            counter_qa_sims = self.cal_qa_aug_score(img, boxs, q, obj_name)
            
            # counter_sims = 0.4 * counter_sim_sims + 0.3 * counter_cap_sims + 0.3 * counter_qa_sims
            counter_sims = counter_qa_sims
            
            tau = 1.2
            norm_counter_scores = torch.softmax(counter_sims, dim=-1)
            norm_counter_scores = [norm_counter_scores[ii] * (tau-1/tau) + 1/tau for ii in range(len(norm_counter_scores))]
            score_weight[:,idx] = torch.tensor(norm_counter_scores)
            
            ct_score += counter_qa_sims.mean().item()
            
        if len(parts): ct_score /= len(parts)
            
        scores = scores * score_weight
        obj_ids = scores.argmax(0)
        parts = list(set([parts[idx] for idx in obj_ids]))
        scores = [scores[i,idx] for idx,i in enumerate(obj_ids)]
        
        return sum(scores)/len(scores), ct_score, parts
    

    def html(self,img_var,obj_var,parts,output_var,output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        part_arg = html_arg_name('part')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{part_arg}={parts})={output}</div>"""


    def execute(self,prog_step,inspect=False):
        img_var,obj_var,part_var,query,output_var = self.parse(prog_step)
        # imgs = prog_step.state[img_var]
        
        # if type(imgs) is not list:
        imgs = prog_step.state["IMAGE"]
        
        objss = prog_step.state[obj_var]
        parts = prog_step.state[part_var]
        scores = []
        
        # # # way1: whole score
        # text = f'a photo of {query} with ' + ','.join(parts)
        # # print(imgs)
        # # similarity
        # inputs = self.processor(
        #     text=text, images=imgs, return_tensors="pt", padding=True).to(self.device)
        # with torch.no_grad():
        #     whole_scores = self.calculate_sim(inputs).squeeze(-1).tolist()
        # # counter factual similarity
        # ct_whole_scores = self.cal_sim_aug_score_whole(imgs, query)
        # minx, maxx = min(ct_whole_scores), max(ct_whole_scores)
        # tau = 1.5
        # normed_whole_scores = [(x-minx)/(maxx-minx)*(tau-1/tau)+1/tau for x in ct_whole_scores]
        # whole_scores = torch.tensor(whole_scores) * torch.tensor(normed_whole_scores)
        
        ct_score = 0
        
        part_scores = []
        predict_parts = []
        for i in range(len(imgs)):
            part = []
            img = imgs[i]
            objs = objss[i]
            if len(objs) <= 1:
                score, ct_score, part = 0, 0, []
            else:
                if parts is not None:
                    # score = self.query_obj(parts, objs, img)
                    score, ct_score, part = self.ct_aug_query_obj(parts, objs, img, query)
                else:
                    score, ct_score, part = 0, 0, []
            # scores.append(score+whole_scores[i])
            part_scores.append(score)
            predict_parts.append(part)

        
        # # way1 aug whole score
        # minx, maxx = min(part_scores), max(part_scores)
        # tau = 1.1
        # normed_part_scores = [(x-minx)/(maxx-minx)*(tau-1/tau)+1/tau for x in part_scores]
        # scores = torch.tensor(whole_scores) * torch.tensor(normed_part_scores)
        
        # # way3 text + parts
        texts = []
        for part in predict_parts:
            if part:
                texts.append(f"a photo of {query} with " + ','.join(part))
            else:
                texts.append(f"a photo of {query}")
        # # similarity
        inputs = self.processor(
            text=texts, images=imgs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            whole_scores = self.calculate_sim(inputs).squeeze(-1).tolist()
            whole_scores = [whole_scores[i][i] for i in range(len(whole_scores))]
        # counter factual similarity
        ct_whole_scores = self.cal_sim_aug_score_whole(imgs, query)
        minx, maxx = min(ct_whole_scores), max(ct_whole_scores)
        tau = 1.5
        normed_whole_scores = [(x-minx)/(maxx-minx)*(tau-1/tau)+1/tau for x in ct_whole_scores]
        whole_scores = torch.tensor(whole_scores) * torch.tensor(normed_whole_scores)
        
        scores = whole_scores
        
        # print(scores)
        index = torch.argmax(torch.tensor(scores)).item()
        prog_step.state[output_var] = index
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            html_str = self.html(img_var, obj_var, parts, output_var, imgs[index])
            return index, html_str

        return index
    
    def cal_sim_aug_score_whole(self, imgs, obj_name):
        text = f'a photo of {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'a photo of {ct_obj}'
        
        # print(text, ct_text)
        
        objs = imgs
        clip_inputs = self.processor(
            text=text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        clip_inputs = self.processor(
            text=ct_text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # ct_sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()    
            ct_sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        return counter_sims
    
    
    def cal_sim_aug_score(self, img, boxes, obj_name, whole_name):
        text = f'a photo of {obj_name} of {whole_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'a photo of {ct_obj} of {whole_name}'
        
        # print(text, ct_text)
        
        objs = [img.crop(o) for o in boxes]
        clip_inputs = self.processor(
            text=text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        clip_inputs = self.processor(
            text=ct_text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # ct_sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()    
            ct_sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        return counter_sims
    
    def cal_cap_aug_score(self, img, boxes, obj_name, whole_name):
        
        # get scores of all candidates: clip_score
        text = f'The part looks like  {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'The part looks like  {ct_obj}'
        
        objs = [img.crop(o) for o in boxes]
        texts = ['The part looks like '] * len(objs)
        instructblip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        text_embedding = self.sent_model.encode(text, convert_to_tensor=True, device=self.device)
        counter_text_embedding = self.sent_model.encode(ct_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            outputs = self.blip_model.generate(**instructblip_inputs)
            responses = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
            # sims = bert_score(responses, [text])
            responses = ['The part looks like '+x for x in responses]
            responses_embedding = self.sent_model.encode(responses, convert_to_tensor=True, device=self.device)
            # responses_embedding_norm = torch.nn.functional.normalize(responses_embedding, p=2, dim=1)
            # b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
            # return torch.mm(a_norm, b_norm.transpose(0, 1))
            responses_sims = util.pytorch_cos_sim(responses_embedding, text_embedding).squeeze(-1)
            ct_responses_sims = util.pytorch_cos_sim(responses_embedding, counter_text_embedding).squeeze(-1)
            
        # print('response', responses_sims)
        # print('counter response', ct_responses_sims)
        counter_sims = responses_sims - ct_responses_sims
        
        return counter_sims
    
    def cal_qa_aug_score(self, img, boxes, obj_name, whole_name):
        
        # get scores of all candidates: clip_score
        text = f'Does this part looks like {obj_name} of {whole_name} ?'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'Does this part looks like {ct_obj} of {whole_name} ?'
        
        objs = [img.crop(o) for o in boxes]
        texts = [text] * len(objs)
        ct_texts = [ct_text] * len(objs)
        blip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        ct_blip_inputs = self.blip_processor(
            text=ct_texts, images=objs, return_tensors='pt'
        ).to(self.device)
        constrained_index = self.blip_processor(text=['yes', 'no'])["input_ids"] # instruct-blip is case-sensitive model
        constrained_index = [x[1] for x in constrained_index]
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**blip_inputs, 
                                                        num_beams=1,
                                                        return_dict_in_generate=True,
                                                        output_scores=True)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            logits = outputs.scores[1]
            contrained_logits = logits[:, constrained_index]
            contrained_logits = torch.softmax(contrained_logits, dim=-1)
            responses_sims =  contrained_logits[:, 0] - contrained_logits[:, 1]
            
            ct_outputs = self.blip_model.generate(**ct_blip_inputs,
                                                            num_beams=1,
                                                            return_dict_in_generate=True,
                                                            output_scores=True)

            ct_logits = ct_outputs.scores[1]
            ct_contrained_logits = ct_logits[:, constrained_index]
            ct_contrained_logits = torch.softmax(ct_contrained_logits, dim=-1)
            ct_responses_sims =  ct_contrained_logits[:, 1] - ct_contrained_logits[:, 0]
        counter_sims = (responses_sims + ct_responses_sims) / 2
        
        return counter_sims
    
    

class FilterInterpreter():
    step_name = 'FILTER'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = CUDAIDX if torch.cuda.is_available() else "cpu"
        # self.model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.model.eval()
        # self.processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        self.model = CLIPModel.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            f"{ckpt_dir}/clip-vit-large-patch14")
        
        
        # # for verification
        # self.clip_model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.clip_model.eval()
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        # self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # self.blip_model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-capfilt-large").to(self.device)
        # self.blip_model.eval()
        # # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        # #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # # self.instructblip_model.eval()
        # self.sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        
        
        self.blip_processor = AutoProcessor.from_pretrained(f"{ckpt_dir}/blip-vqa-capfilt-large")
        self.blip_model = BlipForQuestionAnswering.from_pretrained(
            f"{ckpt_dir}/blip-vqa-capfilt-large").to(self.device)
        self.blip_model.eval()
        # self.instructblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        # self.instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/instructblip-flan-t5-xl").to(self.device) # instructblip-vicuna-7b
        # self.instructblip_model.eval()
        self.sent_model = SentenceTransformer(f'{ckpt_dir}/sentence-transformers_all-MiniLM-L6-v2').to(self.device)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = parse_result['args']['query']
        # query = eval(parse_result['args']['query']).split(',')
        query = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        # text = [f'a photo of {q}' for q in query]
        text = f'a photo of {query}'
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).squeeze(-1).cpu().numpy()
        objs, scores = zip(*sorted(zip(objs, scores), key=lambda x:x[1], reverse=True))
        # obj_ids = scores.argmax(0)
        # return [objs[i] for i in obj_ids]
        return objs
    
    def ct_aug_query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        # text = [f'a photo of {q}' for q in query]
        text = f'a photo of {query}'
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).squeeze(-1).cpu()
            
        boxs = [obj['box'] for obj in objs]
        counter_sim_sims = self.cal_sim_aug_score(img, boxs, query)
        # counter_cap_sims = self.cal_cap_aug_score(img, boxs, query)
        # counter_qa_sims = self.cal_qa_aug_score(img, boxs, query)

        # counter_sims = 0.4 * counter_sim_sims + 0.3 * counter_cap_sims + 0.3 * counter_qa_sims
        # counter_sims = 0.5 * counter_cap_sims + 0.5 * counter_qa_sims
        # counter_sims = counter_qa_sims
        counter_sims = counter_sim_sims

        tau = 1.2
        norm_counter_scores = torch.softmax(counter_sims, dim=-1)
        norm_counter_scores = [norm_counter_scores[ii] * (tau-1/tau) + 1/tau for ii in range(len(norm_counter_scores))]
        
        score_weight = torch.tensor(norm_counter_scores)
        scores = scores * score_weight
        
        objs, scores = zip(*sorted(zip(objs, scores), key=lambda x:x[1], reverse=True))
        # obj_ids = scores.argmax(0)
        # return [objs[i] for i in obj_ids]
        return objs, counter_sims.mean().item()
        

    def html(self,img_var,obj_var,query,category,output_var,output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{query_arg}={query})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,query,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        if len(objs) <= 1: 
            prog_step.state[output_var] = objs
            return objs
        select_objs = []
        
        ct_score = 0

        if query is not None:
            if query == 'LEFT':
                cxs = [(obj['box'][0] + obj['box'][2])/2 for obj in objs]
                sorted_cxs = sorted(cxs)
                median = sorted_cxs[len(sorted_cxs)//2]
                for ii, cx in enumerate(cxs):
                    if cx <= median:
                        select_objs.append(objs[ii])
            elif query == 'RIGHT':
                cxs = [(obj['box'][0] + obj['box'][2])/2 for obj in objs]
                sorted_cxs = sorted(cxs, reverse=True)
                median = sorted_cxs[len(sorted_cxs)//2]
                for ii, cx in enumerate(cxs):
                    if cx >= median:
                        select_objs.append(objs[ii])
            elif query == 'TOP' or query == 'UP':
                cys = [(obj['box'][1] + obj['box'][3])/2 for obj in objs]
                sorted_cys = sorted(cys)
                median = sorted_cys[len(sorted_cys)//2]
                for ii, cy in enumerate(cys):
                    if cy <= median:
                        select_objs.append(objs[ii])
            elif query == 'BOTTOM' or query == 'DOWN':
                cys = [(obj['box'][1] + obj['box'][3])/2 for obj in objs]
                sorted_cys = sorted(cys, reverse=True)
                median = sorted_cys[len(sorted_cys)//2]
                for ii, cy in enumerate(cys):
                    if cy >= median:
                        select_objs.append(objs[ii])
            else:
                # select_objs = self.query_obj(query, objs, img)
                select_objs, ct_score = self.ct_aug_query_obj(query, objs, img)

        prog_step.state[output_var] = select_objs
        prog_step.state['CT_SCORE'] += ct_score
        if inspect:
            select_obj_img = vis_masks(img, select_objs)
            html_str = self.html(img_var, obj_var, query, output_var, select_obj_img)
            return select_objs, html_str
        return select_objs
    
    def cal_sim_aug_score(self, img, boxes, obj_name):
        text = f'a photo of {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'a photo of {ct_obj}'
        
        # print(text, ct_text)
        
        objs = [img.crop(o) for o in boxes]
        clip_inputs = self.processor(
            text=text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()
            sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        clip_inputs = self.processor(
            text=ct_text, images=objs, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            # ct_sims = self.calculate_sim(clip_inputs).squeeze(-1).cpu().numpy()    
            ct_sims = self.calculate_sim(clip_inputs).squeeze(-1)
        
        counter_sims = sims - ct_sims
        
        return counter_sims
    
    def cal_cap_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'an image describe {obj_name}'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'an image describe {ct_obj}'
        
        objs = [img.crop(o) for o in boxes]
        texts = ['an image describe '] * len(objs)
        blip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        text_embedding = self.sent_model.encode(text, convert_to_tensor=True, device=self.device)
        counter_text_embedding = self.sent_model.encode(ct_text, convert_to_tensor=True, device=self.device)
        with torch.no_grad():
            outputs = self.blip_model.generate(**blip_inputs)
            responses = self.blip_processor.batch_decode(outputs, skip_special_tokens=True)
            # sims = bert_score(responses, [text])
            responses = ['an image describe '+x for x in responses]
            responses_embedding = self.sent_model.encode(responses, convert_to_tensor=True, device=self.device)
            # responses_embedding_norm = torch.nn.functional.normalize(responses_embedding, p=2, dim=1)
            # b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
            # return torch.mm(a_norm, b_norm.transpose(0, 1))
            responses_sims = util.pytorch_cos_sim(responses_embedding, text_embedding).squeeze(-1)
            ct_responses_sims = util.pytorch_cos_sim(responses_embedding, counter_text_embedding).squeeze(-1)
            
        # print('response', responses_sims)
        # print('counter response', ct_responses_sims)
        counter_sims = responses_sims - ct_responses_sims
        
        return counter_sims
    
    def cal_qa_aug_score(self, img, boxes, obj_name):
        
        # get scores of all candidates: clip_score
        text = f'Does {obj_name} correcly describe the image ?'
        ct_obj = random.choice(anto_dct.get(obj_name, ['stocking']))
        ct_text = f'Does {ct_obj} correctly describe the image ?'
        
        objs = [img.crop(o) for o in boxes]
        texts = [text] * len(objs)
        ct_texts = [ct_text] * len(objs)
        blip_inputs = self.blip_processor(
            text=texts, images=objs, return_tensors='pt'
        ).to(self.device)
        ct_blip_inputs = self.blip_processor(
            text=ct_texts, images=objs, return_tensors='pt'
        ).to(self.device)
        
        constrained_index = self.blip_processor(text=['yes', 'no'])["input_ids"] # instruct-blip is case-sensitive model
        constrained_index = [x[1] for x in constrained_index]
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**blip_inputs, 
                                                        num_beams=1,
                                                        return_dict_in_generate=True,
                                                        output_scores=True)
            # responses = self.instructblip_processor.batch_decode(outputs, skip_special_tokens=True)
            logits = outputs.scores[1]
            contrained_logits = logits[:, constrained_index]
            contrained_logits = torch.softmax(contrained_logits, dim=-1)
            responses_sims =  contrained_logits[:, 0] - contrained_logits[:, 1]
            
            ct_outputs = self.blip_model.generate(**ct_blip_inputs,
                                                            num_beams=1,
                                                            return_dict_in_generate=True,
                                                            output_scores=True)

            ct_logits = ct_outputs.scores[1]
            ct_contrained_logits = ct_logits[:, constrained_index]
            ct_contrained_logits = torch.softmax(ct_contrained_logits, dim=-1)
            ct_responses_sims =  ct_contrained_logits[:, 1] - ct_contrained_logits[:, 0]
 
        counter_sims = (responses_sims + ct_responses_sims) / 2
        
        return counter_sims
    
    

class PartInterpreter():
    step_name = 'PART'

    prompt_examples = ["""
Query: List parts of table with tablecloth separated by commas
List:
tablecloth,leg""",
"""
Query: List parts of bird flying separated by commas
List:
body,wing,head""",
"""
Query: List parts of dog separated by commas
List:
body,head""",
"""
Query: List parts of teepee separated by commas
List:
tent,door,base""",
"""
Query: List parts of train behind mountains separated by commas
List:
train,mountains"""]

    pre_prompt_template = "Create comma separated lists based on the query."
    post_prompt_template = "Query: List parts of {text} separated by commas \n List:"


    def __init__(self):
        print(f'Registering {self.step_name} step')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        text = eval(parse_result['args']['query'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return text,output_var

    def get_list(self,text):
        # response = openai.Completion.create(
        #     model="text-davinci-002",
        #     prompt=self.prompt_template.format(list_max=list_max,text=text),
        #     temperature=0.7,
        #     max_tokens=256,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1,
        # )
        # item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
        
        # messages=[
        #             {"role": "user", "content": self.prompt_template.format(text=text)}
        #         ]
        # response = openai_api_stream(messages).strip('\n')
        
        self.prompt_template = self.pre_prompt_template + ' \n '.join(random.sample(self.prompt_examples, int(random.uniform(0.7, 1.0)*len(self.prompt_examples)))) + self.post_prompt_template
        
        try:
            cached_parts = 'gpt-3.5-turbo_kilogram_parts.pkl'
            with open(cached_parts, 'rb') as pp:
                cached_parts = pickle.load(pp)
            # response = list(cached_parts['gpt-3.5-turbo'][text])[-1]
            response = random.choice(cached_parts['gpt-3.5-turbo'][-2:])
        except:
            messages=[
                        {"role": "user", "content": self.prompt_template.format(text=text)}
                    ]
            response = openai_api_stream(messages).strip('\n')
        
        
        item_list = response.split(',')
        
        return item_list

    def html(self,text,item_list,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        output = html_output(item_list)
        return f"""<div>{output_var}={step_name}({query_arg}='{text}')={output}</div>"""

    def execute(self,prog_step,inspect=False):
        text,output_var = self.parse(prog_step)
        item_list = self.get_list(text)
        prog_step.state[output_var] = item_list
        if inspect:
            html_str = self.html(text, item_list, output_var)
            return item_list, html_str

        return item_list

    


def register_step_interpreters(dataset='nlvr'):
    if dataset=='nlvr':
        return dict(
            VQA=VQAInterpreter(), # neural
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='gqa':
        return dict(
            LOC=LocInterpreter(), # neural
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(), # neural
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter(),
            # SIM=SimInterpreter(), # neural
            # TEST=TestInterpreter()
        )
    elif dataset=='agqa':
        return dict(
            VQA=VQAInterpreter(), # neural
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter(),
            FIND=FindInterpreter(),
            MEASURE=MeasureInterpreter(),
            GET=GetInterpreter(),
            GET_BEFORE=GetBeforeInterpreter(),
            GET_AFTER=GetAfterInterpreter(),
            GET_BETWEEN=GetBetweenInterpreter(),
            LOC=LocInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
        )
    elif dataset=='magicbrush':
        return dict(
            # FACEDET=FaceDetInterpreter(), # neural
            SEG=SegmentInterpreter(), # neural
            SELECT=SelectInterpreter(), # neural
            # COLORPOP=ColorpopInterpreter(),
            # BGBLUR=BgBlurInterpreter(),
            REPLACE=ReplaceInterpreter(), # neural
            # EMOJI=EmojiInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='refcoco':
        return dict(
            # FACEDET=FaceDetInterpreter(), # neural
            # LIST=ListInterpreter(), # neural
            # CLASSIFY=ClassifyInterpreter(), # neural
            RESULT=ResultInterpreter(), 
            TAG=TagInterpreter(),
            LOC=Loc2Interpreter(thresh=0.05,nms_thresh=0.3), # neural
            FILTER=FilterInterpreter(),
        )
    elif dataset=='kilogram':
        return dict(
            PART=PartInterpreter(),
            SEGS=SegmentsInterpreter(),
            ALIGN=AlignInterpreter(),
            RESULT=ResultInterpreter(),
        )