# -*- coding: utf-8 -*-
from flask import Flask, request, redirect, render_template
import os
import re
import glob
import numpy as np
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib
import pandas as pd
import text_cleaner

# flask
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload/', methods=['POST'])
def upload_multipart():
    file = request.files['file']
    fileName = file.filename
    file.save(os.path.join('tmp', fileName))
    return redirect('/tfidf/')

def parsewithelimination(sentense):
  slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
  slothlib_file = urllib.request.urlopen(slothlib_path)
  stopwords=[]
  for line in slothlib_file:
    ss=line.decode("utf-8").strip()
    if not ss==u'':
      stopwords.append(ss)

  elim=['数','非自立','接尾']
  part=['名詞', '動詞', '形容詞']

  m=MeCab.Tagger()
  m.parse('')
  node=m.parseToNode(sentense)
    
  result=''
  while node:
    if node.feature.split(',')[6] == '*': # 原形を取り出す
      term=node.surface
    else :
      term=node.feature.split(',')[6]
        
    if term in stopwords:
      node=node.next
      continue
    
    if node.feature.split(',')[1] in elim:
      node=node.next
      continue

    if node.feature.split(',')[0] in part:
      if result == '':
        result = term
      else:
        result=result.strip() + ' '+ term
        
    node=node.next

  return result

@app.route('/tfidf/')
def calc_tfidf():
    wakatilist = []
    for filename in glob.glob('./tmp/*.txt'):
        with open(filename, 'r', encoding='shift-jis')as f:
            res = text_cleaner.Cleaner(filename)
            wakatilist.append(parsewithelimination('\n'.join(res.read())))
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, token_pattern=u'(?u)\\b\\w+\\b')
    tfidf = vectorizer.fit_transform(wakatilist)
    itemlist=sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])
    columns = [u[0] for u in itemlist]  # 欄の見出し（単語）を付ける
    df = pd.DataFrame(tfidf.toarray(), index=glob.glob('./tmp/*.txt'), columns=columns)
    df_values = df.values.tolist()
    df_columns = df.columns.tolist()
    df_index = df.index.tolist()
    return render_template('tfidf.html', \
        df_values = df_values, \
        df_columns = df_columns, \
        df_index = df_index)


## おまじない
if __name__ == "__main__":
    app.run(debug=True)
