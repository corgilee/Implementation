# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:44:45 2019

@author: Wenjie
"""
'''
https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
'''


import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import math


text1="""
if you like tuna and tomato sauce- try combining the two.
it's really not as bad as it sounds.
if the Eater Bunny and the Tooth Fairy had babies would they take
your teetch and leave chocolate for you?
"""

def remove_string_special_characters(s):
    """
    This funciton removes special characters from within a string
    """
    stripped=re.sub('[^\w\s]','',s) #stripped means special sign
    stripped=re.sub("_","",stripped)
    
    #change any whitespace to one space
    stripped=re.sub("\s+"," ",stripped)
    
    #remove start and end white spaces
    #strip() in-built function of Python is used to remove all the leading and trailing spaces from a string.
    stripped=stripped.strip()
    
    return stripped

def count_words(sent):
    '''This function returns the total number of words in the input test.
    '''
    count=0
    words=word_tokenize(sent)
    for word in words:
        count+=1
    return count

'Considering each sentence as a document,calculates the total word count of each.'''
def get_doc(sent):
    doc_info=[]
    i=0
    for sent in text_sents_clean:
        i+=1
        count=count_words(sent)
        temp={'doc_id':i, 'doc_length':count}
        doc_info.append(temp)
    return doc_info

def create_freq_dict(sents):
    '''
    This function creates a frequency dictionary for each word in each document.
    '''
    i=0
    freqDict_list=[]
    for sent in sents:
        i+=1
        freq_dict={}
        words=word_tokenize(sent)
        for word in words:
            word=word.lower()
            if word in freq_dict:
                freq_dict[word]+=1
            else:
                freq_dict[word]=1
        temp={"doc_id":i,"freq_dict":freq_dict}
        freqDict_list.append(temp)
    return freqDict_list


#The functions to get the TF and IDF score:

def computeTF(doc_info,freqDict_list):
    '''
    tf=(frequency of the term in the doc/total number of terms in the doc)
    Term Frequency 
    '''
    TF_scores=[]
# freqDict_list now has several freq_dict   
    for tempDict in freqDict_list:
        id=tempDict['doc_id']
        for k in tempDict["freq_dict"]:
            temp={"doc_id":id,
                  "TF_score": tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],
                  "key":k}
            TF_scores.append(temp)
    return TF_scores

def computeIDF(doc_info,freqDict_list):
    '''
    idf=ln(total number of docs/number of docs with term in it)
    '''
    IDF_scores=[]
    counter=0
    for dict in freqDict_list:
        counter+=1
        for k in dict["freq_dict"].keys():
            count=sum([k in tempDict['freq_dict'] for tempDict in freqDict_list ])
            temp={"doc_id":counter,"IDF_score": math.log(len(doc_info)/count),"key":k}
            IDF_scores.append(temp)
    return IDF_scores


text_sents=sent_tokenize(text1)  
text_sents_clean=[remove_string_special_characters(s) for s in text_sents]  
doc_info=get_doc(text_sents_clean)

freqDict_list=create_freq_dict(text_sents_clean)
TF_scores=computeTF(doc_info,freqDict_list)
IDF_scores=computeIDF(doc_info,freqDict_list)


#TF*IDF
def computeTFIDF(TF_scores,IDF_scores):
    TFIDF_scores=[]
    for j in IDF_scores:
        for i in TF_scores:
            if j["key"]==i["key"] and j["doc_id"]==i["doc_id"]:
                temp={"doc_id":j["doc_id"],
                      "TFIDF_score":j["IDF_score"]*i["TF_score"],
                      "key":i["key"]}
        TFIDF_scores.append(temp)
    return TFIDF_scores

#print(doc_info)
#print(freqDict_list)
#print(TF_scores)
#print(IDF_scores)
   
TFIDF_scores=computeTFIDF(TF_scores,IDF_scores)
print(TFIDF_scores)