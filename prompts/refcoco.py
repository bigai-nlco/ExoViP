import random

REFCOCOG_CURATED_EXAMPLES = [
"""
Instruction: Tag a white woman with a brown ponytail wearing a red tank top crouched and washing the foot of a goat
Program:
OBJ0=LOC(image=IMAGE,object='woman')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='woman with a brown ponytai')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='woman wearing a red tank top crouched')
OBJ3=FILTER(image=IMAGE,object=OBJ2,query='woman washing the foot of a goat')
IMAGE0=TAG(image=IMAGE,object=OBJ3)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag a black computer screen
Program:
OBJ0=LOC(image=IMAGE,object='screen')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='black computer screen')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag a white cow next to a brown cow
Program:
OBJ0=LOC(image=IMAGE,object='cow')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='a white cow')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag far left car
Program:
OBJ0=LOC(image=IMAGE,object='car')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query="LEFT")
IMAGE1=TAG(image=IMAGE0,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE1)
""",
"""
Instruction: Tag an elephant walks among boulders and dirt in an outdoor setting
Program:
OBJ0=LOC(image=IMAGE,object='elephant')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='an elephant walks among boulders and dirt in an outdoor setting')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag a beautiful hen
Program:
OBJ0=LOC(image=IMAGE,object='hen')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='a beautiful hen')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag second donut from the bottom right
Program:
OBJ0=LOC(image=IMAGE,object='donut')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='BOTTOM')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='RIGHT')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag dude in blue jeans white and brown shirt
Program:
OBJ0=LOC(image=IMAGE,object='person')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='person in blue jeans')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='person in white and brown shirt')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag a mini refridgator with a lot of stickers on the front
Program:
OBJ0=LOC(image=IMAGE,object='refridgator')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='a mini refridgator with a lot of stickers on the front')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag a silver motorcycle on the right
Program:
OBJ0=LOC(image=IMAGE,object='motorcycle')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='RIGHT')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
"""
]


REFCOCO_CURATED_EXAMPLES = [
"""
Instruction: Tag guy holding purp umbrella in corner near us
Program:
OBJ0=LOC(image=IMAGE,object='guy')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='guy holding purp umbrella')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag burried donut plain at 2 o clock
Program:
OBJ0=LOC(image=IMAGE,object='donut')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='BOTTOM')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='TOP')
OBJ3=FILTER(image=IMAGE,object=OBJ2,query='burried donut plain')
IMAGE0=TAG(image=IMAGE,object=OBJ3)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag woman smiling in tan sweater
Program:
OBJ0=LOC(image=IMAGE,object='woman')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='woman smiling in tan sweater')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag far left car
Program:
OBJ0=LOC(image=IMAGE,object='car')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query="LEFT")
IMAGE1=TAG(image=IMAGE0,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE1)
""",
"""
Instruction: Tag man
Program:
OBJ0=LOC(image=IMAGE,object='man')
IMAGE0=TAG(image=IMAGE,object=OBJ0)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag red shirt
Program:
OBJ0=LOC(image=IMAGE,object='shirt')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='red shirt')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag second donut from the bottom right
Program:
OBJ0=LOC(image=IMAGE,object='donut')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='BOTTOM')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='RIGHT')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag dude in blue jeans white and brown shirt
Program:
OBJ0=LOC(image=IMAGE,object='person')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='person in blue jeans')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='person in white and brown shirt')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag number 8
Program:
OBJ0=LOC(image=IMAGE,object='8')
IMAGE0=TAG(image=IMAGE,object=OBJ0)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Instruction: Tag girls hair
Program:
OBJ0=LOC(image=IMAGE,object='hair')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='girls hair')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
"""
]

PROMPT = """Think step by step to carry out the instruction.

Instruction: Tag guy holding purp umbrella in corner near us
Program:
OBJ0=LOC(image=IMAGE,object='guy')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='guy holding purp umbrella')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag burried donut plain at 2 o clock
Program:
OBJ0=LOC(image=IMAGE,object='donut')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='BOTTOM')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='TOP')
OBJ3=FILTER(image=IMAGE,object=OBJ2,query='burried donut plain')
IMAGE0=TAG(image=IMAGE,object=OBJ3)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag woman smiling in tan sweater
Program:
OBJ0=LOC(image=IMAGE,object='woman')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='woman smiling in tan sweater')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag far left car
Program:
OBJ0=LOC(image=IMAGE,object='car')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query="LEFT")
IMAGE1=TAG(image=IMAGE0,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE1)

Instruction: Tag man
Program:
OBJ0=LOC(image=IMAGE,object='man')
IMAGE0=TAG(image=IMAGE,object=OBJ0)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag red shirt
Program:
OBJ0=LOC(image=IMAGE,object='shirt')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='red shirt')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag second donut from the bottom right
Program:
OBJ0=LOC(image=IMAGE,object='donut')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='BOTTOM')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='RIGHT')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag dude in blue jeans white and brown shirt
Program:
OBJ0=LOC(image=IMAGE,object='person')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='person in blue jeans')
OBJ2=FILTER(image=IMAGE,object=OBJ1,query='person in white and brown shirt')
IMAGE0=TAG(image=IMAGE,object=OBJ2)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag number 8
Program:
OBJ0=LOC(image=IMAGE,object='8')
IMAGE0=TAG(image=IMAGE,object=OBJ0)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Tag girls hair
Program:
OBJ0=LOC(image=IMAGE,object='hair')
OBJ1=FILTER(image=IMAGE,object=OBJ0,query='girls hair')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {instruction}
Program:
"""

def create_prompt(inputs, num_prompts=3, method='random',seed=42,group=0):
    if method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(REFCOCO_CURATED_EXAMPLES,num_prompts)
    elif method == 'all':
        prompt_examples = REFCOCO_CURATED_EXAMPLES
    else:
        raise NotImplementedError
    
    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f"Think step by step to carry out the instruction.\n\n{prompt_examples}"
    return prompt_examples + "\nInstruction: {instruction}\nProgram:".format(**inputs)