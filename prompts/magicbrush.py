import random

MAGICBRUSH_CURATED_EXAMPLES = [
"""
Instruction: Change the table for a dog
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='table',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='dog')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: What if it was another car on it
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='car',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='car')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Put a towel on the hanger
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='hanger',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='towel on the hanger')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Remove the people
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='people',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='no people')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Make the stove be on fire
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='stove',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='stove on fire')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Let's add a apple next to the microwave
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='microwave',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='microwave and apple')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Change the bus color to blue
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Replace the red bus with blue bus and the road with dirt road
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
OBJ2=SEG(image=IMAGE0)
OBJ3=SELECT(image=IMAGE0,object=OBJ2,query='road',category=None)
IMAGE1=REPLACE(image=IMAGE0,object=OBJ3,prompt='dirt road')
FINAL_RESULT=RESULT(var=IMAGE1)
""",
"""
Instruction: Have the woman be holding a trophy
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='woman',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='woman holding a trophy')
FINAL_RESULT=RESULT(var=IMAGE0)
"""
]

PROMPT = """Think step by step to carry out the instruction.

Instruction: Change the table for a dog
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='table',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='dog')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: What if it was another car on it
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='car',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='car')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Put a towel on the hanger
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='hanger',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='towel on the hanger')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Remove the people
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='people',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='no people')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Make the stove be on fire
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='stove',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='stove on fire')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Let's add a apple next to the microwave
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='microwave',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='microwave and apple')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Change the bus color to blue
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Replace the red bus with blue bus and the road with dirt road
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
OBJ2=SEG(image=IMAGE0)
OBJ3=SELECT(image=IMAGE0,object=OBJ2,query='road',category=None)
IMAGE1=REPLACE(image=IMAGE0,object=OBJ3,prompt='dirt road')
FINAL_RESULT=RESULT(var=IMAGE1)

Instruction: Have the woman be holding a trophy
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='woman',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='woman holding a trophy')
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {instruction}
Program:
"""

def create_prompt(inputs, num_prompts=7, method='random',seed=42,group=0):
    if method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(MAGICBRUSH_CURATED_EXAMPLES,num_prompts)
    elif method == 'all':
        prompt_examples = MAGICBRUSH_CURATED_EXAMPLES
    else:
        raise NotImplementedError
    
    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f"Think step by step to carry out the instruction.\n\n{prompt_examples}"
    return prompt_examples + "\nInstruction: {instruction}\nProgram:".format(**inputs)