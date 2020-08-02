#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 01:02:14 2020

@author: shyambhu.mukherjee
"""

import pdfminer
import io
from pdfminer.layout import LAParams
from pdfminer import high_level
from pdfminer.high_level import extract_text_to_fp as extex

def extract_raw_text(pdf_filename):
    output = io.StringIO()
    laparams = LAParams() 
    # Using the defaults seems to work fine

    with open(pdf_filename, "rb") as pdffile:
        extex(pdffile, output, laparams=laparams)

    return output.getvalue()
text1 = ""
#text1 = extract_raw_text('/home/shyambhu.mukherjee/Downloads/duckett.pdf')
text2 = extract_raw_text('/home/shyambhu.mukherjee/Downloads/html_tutorial.pdf')
text_collated = text1 + text2

f = open('html_corpus.txt','w')
f.write(text_collated)
f.close()

def summary_utils(summary):
    summar = summary[0]['summary_text']
    return summar

f = open('html_corpus.txt','r')
fultext = f.read()
#bart-large-cnn model
import transformers
from transformers import pipeline
qaObj = pipeline('question-answering')
ans = qaObj(question = "write html code for red button",context = fultext)
"""
{'score': 0.05844178717584647,
 'start': 56248,
 'end': 56269,
 'answer': '405 Language Codes:'}
"""
ans = qaObj(question = "create button",context = fultext)
"""
{'score': 0.9545003905971328,
 'start': 74361,
 'end': 74385,
 'answer': 'height Numeric Value'}
"""
ans = qaObj(question = "write paragraph tag",context = fultext)
"""
same answer as before.
"""
ans = qaObj(question = "paragraph tag",context = fultext)

## code example from w3school
small_text = r"""A paragraph always starts on a new line, and is usually a block of text.
                 HTML Paragraphs
                 The HTML <p> element defines a paragraph.
                 A paragraph always starts on a new line, and browsers automatically add some white space (a margin) before and after a paragraph.
                 Example <p>This is a paragraph.</p>
                 <p>This is another paragraph.</p>
                 HTML Display
                 You cannot be sure how HTML will be displayed.
                 Large or small screens, and resized windows will create different results.
                 With HTML, you cannot change the display by adding extra spaces or extra lines in your HTML code.
                 The browser will automatically remove any extra spaces and lines when the page is displayed:
                 Example <p> This paragraph contains a lot of lines in the source code, 
                 but the browser ignores it. </p> <p> This paragraph contains 
                 a lot of spaces in the source code, but the browser ignores it.</p> 
              """
             
ans = qaObj(question = "write paragraph tag", context = small_text)
"""{'score': 0.27491484847131886,
 'start': 90,
 'end': 131,
 'answer': 'HTML Paragraphs The HTML'}
"""
ans = qaObj(question = "what element is a paragraph tag", context = small_text)
"""
{'score': 0.5403857393915423, 'start': 127, 'end': 131, 'answer': 'HTML'}
"""
ans = qaObj(question = "what defines a pararaph",context = small_text)
"""
{'score': 0.47865264802688756,
 'start': 90,
 'end': 143,
 'answer': 'HTML Paragraphs The HTML <p> element'}
"""

geeks_text = open('geeks_para.txt','r').read()
ans = qaObj(question = "what defines a paragraph",context = geeks_text)
"""
{'score': 0.8452085668869813, 'start': 24, 'end': 28, 'answer': 'HTML'}
"""
ans = qaObj(question = "what tag defines a paragraph",context = geeks_text)
"""
{'score': 0.864106661039159, 'start': 24, 'end': 28, 'answer': 'HTML'}
"""
ans = qaObj(question = "what is a paragraph tag?",context = geeks_text)
"""
{'score': 0.12643090688080516,
 'start': 61,
 'end': 90,
 'answer': 'both opening and closing tag.'}
"""