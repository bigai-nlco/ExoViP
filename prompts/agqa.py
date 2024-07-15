import random

AGQA_CURATED_EXAMPLES=[
"""Question: Was the person opening something before or after snuggling with the object they were carrying?
Program:
IDX0=FIND(image=IMAGE,query='the person is opening something')
IDX1=FIND(image=IMAGE,query='the person is snuggling with the object they were carrying')
ANSWER0=EVAL(expr="'before' if {IDX0}[0] < {IDX1}[0] else 'after'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Was the person closing something before or after turning off a light?
Program:
IDX0=FIND(image=IMAGE,query='the person is closing something')
IDX1=FIND(image=IMAGE,query='the person is turning off a light')
ANSWER0=EVAL(expr="'before' if {IDX0}[0] < {IDX1}[0] else 'after'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Was the person opening a door before or after playing with the thing they put down?
Program:
IDX0=FIND(image=IMAGE,query='the person is opening a door')
IDX1=FIND(image=IMAGE,query='the person is playing with the thing they put down')
ANSWER0=EVAL(expr="'before' if {IDX0}[0] < {IDX1}[0] else 'after'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Was the person opening the object they were holding last before or after holding the thing they went in front of?
Program:
IDX0=FIND(image=IMAGE,query='the person was opening the object they were holding')
IDX1=FIND(image=IMAGE,query='the person was holding the thing they went in front of')
ANSWER0=EVAL(expr="'before' if {IDX0}[0] < {IDX1}[0] else 'afater'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: While holding the last thing they leaned on, did they snuggle anything?
Program:
IDX0=FIND(image=IMAGE,query='they are holding the last thing they leaned on')
IMAGE0=GET(image=IMAGE,pos=IDX0)
ANSWER0=VQA(image=IMAGE0,question='did they snuggle anything?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Was the person watching the thing they stood on while watching television?
Program:
IDX0=FIND(image=IMAGE,query='the person is watching television')
IMAGE0=GET(image=IMAGE,pos=IDX0)
ANSWER0=VQA(image=IMAGE0,question='was the person watching the thing they stood on')
FINAL_RESULT=RESULT(var=ANSER0)
""",
"""Question: Did they watch television for more time than they spent lying on a bed?
Program:
IDX0=FIND(image=IMAGE,query='they are watching television')
IDX1=FIND(image=IMAGE,query='they are lying on a bed')
LENGTH0=MEASURE(pos=IDX0)
LENGTH1=MEASURE(pos=IDX1)
ANSWER0=EVAL(expr="'yes' if {LENGTH0} > {LENGTH1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Did they hold a dish or drink from a cup for less time?
Program:
IDX0=FIND(image=IMAGE,query='they hold a dish')
IDX1=FIND(image=IMAGE,query='they hold a drink from a cup')
LENGTH0=MEASURE(pos=IDX0)
LENGTH1=MEASURE(pos=IDX1)
ANSWER0=EVAL(expr="'holding a dish' if {LENGTH0} < {LENGTH1} else 'holding drink from a cup'")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: In the video, which object were they sitting on?
Program:
IDX0=FIND(image=IMAGE,query='people were sitting')
IMAGE0=GET(image=IMAGE,pos=IDX0)
BOX0=LOC(image=IMAGE0,object='person')
IMAGE1=CROP_BELOW(image=IMAGE0,box=BOX0)
ANSWER0=VQA(image=IMAGE1,question=what is the object?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: While putting the object they were in somewhere, which object did the person go in front of?
Program:
IDX0=FIND(image=IMAGE,query='people were putting the object they were in somewhere')
IMAGE0=GET(image=IMAGE,pos=IDX0)
BOX0=LOC(image=IMAGE0,object='person')
IMAGE1=CROP_BEHIND(image=IMAGE0,box=BOX0)
ANSWER0=VQA(image=IMAGE1,question='what is the object?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: After putting the object they were on the side of last somewhere, what did they start to do first?
Program:
IDX0=FIND(image=IMAGE,query='they were putting the object they were on the side of last somewhere')
IMAGE0=GET_AFTER(image=IMAGE,pos=IDX0)
ANSWER0=VQA(image=IMAGE0,question='what did they start to do first?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: After sitting on the thing they stood on, what is the thing they did?
Program:
IDX0=FIND(image=IMAGE,query='they were sitting on the thing they stood on')
IMAGE0=GET_AFTER(image=IMAGE,pos=IDX0)
ANSWER0=VQA(image=IMAGE0,question='what is the thing they did?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Before putting something on a table but after playing with a phone, which object were they on the side of last?
Program:
IDX0=FIND(image=IMAGE,query='they put something on a table')
IDX1=FIND(image=IMAGE,query='they play with a phone')
IAMGE0=GET_BETWEEN(image=IMAGE,pos1=IDX0,pos2=IDX1)
ANSWER0=VQA(image=IMAGE0,question='which object were they on the side of last?')
FINAL_RESULT=RESULT(var=ANSWER0)
"""]

def create_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='all':
        prompt_examples = AGQA_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(AGQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'


    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)

def create_cr_prompt(inputs,num_prompts=8,method='random',seed=42,group=0):
    if method=='all':
        prompt_examples = AGQA_CURATED_EXAMPLES
    elif method=='random':
        random.seed(seed)
        prompt_examples = random.sample(AGQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f"""Think step by step to answer the question. while taking rejected solutions into account and learning from them.
    Considering the examples provided:\n\n
    ###{prompt_examples}\n\n###
    Here are evaluated solutions that were rejected:
    """

    return prompt_examples + "Here are evaluated solutions that were rejected: ###{rejected_solutions}###".format(**inputs) + "\nQuestion: {question}\nProgram:".format(**inputs)