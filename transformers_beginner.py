#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:48:39 2020

@author: shyambhu.mukherjee
"""

import transformers
from transformers import pipeline
f = open('micro_soft.txt','r')
text = f.read()

def summary_utils(summary):
    summar = summary[0]['summary_text']
    return summar

##abstract summarizer

#T5 model
summarizer = pipeline('summarization',model = "t5-base")
summary_t5_20 = summarizer(text,min_length = 5, max_length = 20)
summary_t10_30 = summarizer(text,min_length = 10, max_length = 30)
summary_t_bigger = summarizer(text,min_length = 100, max_length = 150)

print(text)
print(summary_utils(summary_t5_20))
print(summary_utils(summary_t10_30))
print(summary_utils(summary_t_bigger))

'''
summary_t_create = summarizer(text, min_length = 500, max_length = 1000)
[{'summary_text': 'Microsoft has launched Intelligent Cloud Hub to empower 
 students with AI-ready skills . the three-year collaborative program will 
 support around 100 institutions with AI infrastructure, course content and 
 curriculum, developer support, development tools and give students access 
 to cloud and AI services . as part of the program, the redmond giant will 
 set up the core AI infrastructure and IoT Hub for the selected campuses . 
 Microsoft will also provide Azure AI services such as Microsoft Cognitive 
 Services, Bot Services and Azure Machine Learning . earlier this year, the 
 company announced Microsoft Professional Program In AI .   . -  n    -  . 
 .. -  .-  h  n   . an  an  .    s  "  ,   . " .s  t  - "  "  " " " s " " " 
 ""  "" """ " "- " ".
  " " " -- "--- -, " , ""- & ., . and ./- .&--/--"--\'- s--.--&-. &.-...-/.
  - ".. "- d. /// / &//-// "//"/- "/ " / " "/" "i" "\' " "ii"- "i\' "-" " &
  " "& " \'" "& && \'& ," \'\' "&&\' " c" &###&&&#&#" "## "&/##"&#\' "#& 
  /&&/&# /# & ### -& -/ "&# #&& "& $##\'&& # ##&\'&#/& n&#;&& $&# $&&-&& 
  (&&( &-/&/-&/ -#&/.&#'}]
'''

#bart-large-cnn model

summarizer = pipeline('summarization',model = "facebook/bart-large-cnn")
summary_t5_20 = summarizer(text,min_length = 5, max_length = 20)
summary_t10_30 = summarizer(text,min_length = 10, max_length = 30)
summary_t_bigger = summarizer(text,min_length = 100, max_length = 150)

print(text)
print(summary_utils(summary_t5_20))
print(summary_utils(summary_t10_30))
print(summary_utils(summary_t_bigger))
"""
summary_t_create = summarizer(text, min_length = 500, max_length = 1000)
print(summary_utils(summary_t_create))

summary_t_create = summarizer(text, min_length = 500, max_length = 1000)
print(summary_utils(summary_t_create))
Your max_length is set to 1000, but you input_length is only 368. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)
Intelligent Cloud Hub will support around 100 institutions with AI infrastructure,
course content and curriculum. Redmond giant wants to expand its reach and 
is planning to build a strong developer ecosystem in India with the program. 
The company will provide AI development tools and Azure AI services such as 
Microsoft Cognitive Services, Bot Services and Azure Machine Learning. The 
program aims to build up the cognitive skills and in-depth understanding of 
developing intelligent cloud connected solutions for applications across 
industry. It is an attempt to ramp up the institutional set-up and build 
capabilities among the educators to educate the workforce of tomorrow, 
said Manish Prakash, Country General Manager-PS, Health and Education, 
Microsoft India, in a blog post on Monday. He added that the jobs of 
tomorrow will require a different skillset. This will require more 
collaborations and training and working with AI, he said. 
It's become more critical than ever for educational institutions to 
integrate new cloud and AI technologies, he added, adding that the 
program is a three-year collaborative program that will be rolled out 
over a period of time. It will be open to the public and will be funded by 
the Microsoft Ventures Fund, a venture capital fund that has been set up by
the Redmond giant to invest in the Indian education market. It has been launched 
to support the next generation of students with AI-ready skills. 
It was launched to provide job ready skills to programmers who wanted to hone their skills 
in AI and data science with a series of online courses which featured hands-on labs and expert 
instructors as well. This program also included developer-focused AI school 
that provided a bunch of assets to help build AI skills. Earlier in April 
this year, the company announced Microsoft Professional Program In AI as a 
learning track open toThe company will set up the core AI infrastructure and IoT Hub for the selected campuses. 
It also plans to launch an AI-focused developer school in India in the next few months 
to help students develop their skills for the digital economy. It aims to provide developers 
with a range of tools and development tools to help them develop their own AI-based apps 
and services for the cloud, including Watson and Microsoft Watson, among other tools. 
The project is expected to be completed by the end of this year and will cost around 
$100 million. The first phase of the program will run for three years, 
with the second and third years to be funded over the course of two years. 
The third year will be a two-year program, with an option to extend the program for a further two years if necessary.
"""

summarizer = pipeline('summarization',model = "sshleifer/distilbart-cnn-12-6",
                      tokenizer = "sshleifer/distilbart-cnn-12-6")
summary_t5_20 = summarizer(text,min_length = 5, max_length = 20)
summary_t10_30 = summarizer(text,min_length = 10, max_length = 30)
summary_t_bigger = summarizer(text,min_length = 100, max_length = 150)

print(text)
print(summary_utils(summary_t5_20))
print(summary_utils(summary_t10_30))
print(summary_utils(summary_t_bigger))

summary_t_great = summarizer(text,min_length = 500, max_length = 1000)

#examine the pretrainedtokenizer class of T5
from transformers import T5Tokenizer as TT
tokenizer = TT.from_pretrained("t5-base")
tokens = tokenizer.encode(text,max_length = 512)
decoded_strings = tokenizer.decode(tokens)
decoded_strings
