#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 03:03:25 2020

@author: shyambhu.mukherjee
"""


import transformers
from transformers import pipeline

#gpt2 model
generator = pipeline('text-generation',model = "gpt2",
                     tokenizer="gpt2")
story_50 = generator("once upon a time there was a boy",
                      max_length = 50)
story_100 = generator("once upon a time there was a boy",
                      max_length = 100)
story_1000 = generator("once upon a time there was a boy",
                      max_length = 1000)

print(story_50)
print(story_100)
print(story_1000)
"""
Setting `pad_token_id` to 50256 (first `eos_token_id`) to generate sequence
Setting `pad_token_id` to 50256 (first `eos_token_id`) to generate sequence
[{'generated_text': "once upon a time there was a boy on 
Christmas Day, but the boy's parents 
couldn't find it. It still seemed as if, with the help of a 
little word game, a person of goodwill could be found and the 
person who once tried"}]
[{'generated_text': 'once upon a time there was a boy called John. 
A child, John had lived with his father in the forest, 
and when he had met William he gave him a long-awaited return to the 
city. Though far from alone among his friends, he was a strong 
soldier, and to do more for them they needed a leader. So, William 
put out a message he made in his memory to his father, asking his 
father to help him find his missing friends. For as long as he 
lived'}]
"""

print(story_1000)
"""
NotImplementedError: Generation is currently not supported for 
T5ForConditionalGeneration. Please select a model from 
['XLNetLMHeadModel', 'TransfoXLLMHeadModel', 
'ReformerModelWithLMHead', 'GPT2LMHeadModel', 'OpenAIGPTLMHeadModel', 
'CTRLLMHeadModel', 'TFXLNetLMHeadModel', 'TFTransfoXLLMHeadModel', 
'TFGPT2LMHeadModel', 'TFOpenAIGPTLMHeadModel', 'TFCTRLLMHeadModel'] 
for generation.

Exception: Impossible to guess which tokenizer to use. Please 
provided a PretrainedTokenizer class or a path/identifier to a 
pretrained tokenizer.

OSError: Can't load config for 'GPT2LMHeadModel'. Make sure that:

- 'GPT2LMHeadModel' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'GPT2LMHeadModel' is the correct path to a directory containing a config.json file

"""