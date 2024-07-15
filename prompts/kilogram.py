import random

KILOGRAM_CURATED_EXAMPLES=[
"""
Instruction: flower
Program:
PARTS0=PART(query='flower')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='flower')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: man in robe
Program:
PARTS0=PART(query='man in robe')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='man in robe')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: a claw trying to grab something
Program:
PARTS0=PART(query='a claw trying to grab something')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='a claw trying to grab something')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: pigeon
Program:
PARTS0=PART(query='pigeon')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='pigeon')
FINAL_RESULT=RESULT(var=IMAGE0)
""",    
"""
Instruction: lama
Program:
PARTS0=PART(query='lama')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='lama')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: person handing something off
Program:
PARTS0=PART(query='person handing something off')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='person handing something off')
FINAL_RESULT=RESULT(var=IMAGE0)
""",    
"""
Instruction: table with tablecloth
Program:
PARTS0=PART(query='table with tablecloth')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='table with tablecloth')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: box with food
Program:
PARTS0=PART(query='box with food')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='box with food')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
]

PROMPT = """Think step by step to carry out the instruction.

Instruction: flower
Program:
PARTS0=PART(query='flower')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='flower')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: man in robe
Program:
PARTS0=PART(query='man in robe')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='man in robe')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: a claw trying to grab something
Program:
PARTS0=PART(query='a claw trying to grab something')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='a claw trying to grab something')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: pigeon
Program:
PARTS0=PART(query='pigeon')
OBJ0=SEGS(image=IMAGE)
IMAGE0=ALIGN(image=IMAGE,object=OBJ0,part=PARTS0,query='pigeon')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {statement}
Program:
"""

def create_prompt(inputs, num_prompts=3, method='random',seed=42,group=0):
    if method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(KILOGRAM_CURATED_EXAMPLES,num_prompts)
    elif method == 'all':
        prompt_examples = KILOGRAM_CURATED_EXAMPLES
    else:
        raise NotImplementedError
    
    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f"Think step by step to carry out the instruction.\n\n{prompt_examples}"
    return prompt_examples + "\nInstruction: {instruction}\nProgram:".format(**inputs)