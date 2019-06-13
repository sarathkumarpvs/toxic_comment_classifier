from flask import Flask,request
import pandas as pd, numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

import re, string
app = Flask(__name__)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text=request.form['text']
        return classify(text)
    return '''<!doctype html>
    <title>Toxic comment classifier</title>
    <h1>Enter your comment</h1>
    <form method=post enctype=multipart/form-data>
      <input type=text name=text>
      <input type=submit value=check>
    </form>
    '''
def classify(sen) :
    vec1=pickle.load(open('C:\\Users\\sarathkumar.selvam\\Desktop\\pkl files\\tra.sav', 'rb'))
    sent=pd.DataFrame()
    l1=pickle.load(open('C:\\Users\\sarathkumar.selvam\\Desktop\\pkl files\\mul.sav', 'rb'))
    res={}
    sent['data']=pd.Series(sen)
    print(sent)
    test_x=vec1.transform(sent['data'])
    print(test_x)
    for i in os.listdir('C:\\Users\\sarathkumar.selvam\\Desktop\\pkl files'):
        if 'pkl' in i:
            x=pickle.load(open('C:\\Users\\sarathkumar.selvam\\Desktop\\pkl files\\'+i, 'rb'))
            res[i[:len(i)-4]]=x.predict_proba(test_x.multiply(l1[i[:len(i)-4]]))[:,1][0]
    k=list(res.keys())
    v=list(res.values())
    out=k[v.index(max(v))]
    max_l=[]
    for i in v:
        if i>0.5:
            max_l.append(k[v.index(i)])
            
    if max(v)<0.5:
        return "Your comment does not violate the guidelines"
    else:
        return "Your comment is "+str(max_l)
    
    

    
if __name__ == '__main__':
    app.debug = True
    app.run()
