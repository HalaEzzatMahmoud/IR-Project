import nltk
import os
import math
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import webtext
from pathlib import Path
import statistics
from statistics import mode
import numpy as np
from numpy.linalg import norm
#from sklearn import preprocessing
#nltk.download()

  ## Function For Term Frequency
def TF(sortedWord,wordslist):
    list1 = []
    list2 = []
     #loob over a list of  a sorted words  & x equal to a word in sortedword
    for x in sortedWord:
        list3 = []
        counter = 0
         # loob over words after tokenization & g equal to a word in wordslist "read raw"
        for g in wordslist:
            list4 = []
            counter = counter + 1
            z = 0
              # loob over wordlist lenth & z equal to an incremented number starts with 0 to loob over every word in g "g = raw"
            for j in range(0,len(g)):
                   ## x is the word in sorted word && y[j] is the word in the file
                if x == g[j]:
                    z = z + 1   #every time the word is found increment the counter by 1

            if z==0: ## the word is not found
                list4.append(("doc {}".format(counter),0)) ## its term frequency is 0
            else: ## the word is found
                list4.append(("doc {}".format(counter),1+(math.log10(z)))) ## applay TF law on z => 1+(math.log10(z)
            if list4: ## true
                list3.append(list4) 
        list2.append((x,list3))
    list1.append(list2)
    return list1

#C:\Users\User\Desktop\IRProject\Ir.py
def Calc_IDF(wordslist, positinal_index):
        N=len(wordslist) ## lenth for every word after tokenization
        list1 =[]
        list2 = []
        for y in positinal_index: ## y means entering the pos index list 
                count = 0
                for n in y[2]: ## y[2] where the doc & th position of the word 
                    count+=1
                list1.append([y[0],y[1],math.log10(N/count)])
                
        return list1
  ## function for Tokenization
def Tokens(txt):
  stop_words = set(stopwords.words('english')) ## list of all the stop words in english
    ### Remove these words from stopwords list
  stop_words.remove('where')
  stop_words.remove('in')  
  stop_words.remove('to')
  list3=['.',',',"''"]
  list1 = []
    ## word_tokenize : will break the sentence into words
  words = word_tokenize(txt)
  ## loob over words to check if it is a stopword or not  
  for x in words:
        if str.lower(x) not in stop_words and str.lower(x) not in list3 :
            list1.append(str.lower(x))
  return list1 
  

def countWord(sortedWord, wordslist):
    list1=[]
    #loob over a list of  a sorted words  & x equal to a word in sortedword
    for x in sortedWord:
        #counter for the word
        counter = 0;
         # loob over words after tokenization & g equal to a word in wordslist "read raw"
        for g in wordslist:
          # loob over wordlist lenth & z equal to an incremented number starts with 0 to loob over every word in g "g = raw"
            for z in range(0,len(g)):
              # if word insortedlist "x" in lowercase EQUAL to a word in the file "g[z] = word in the raw"
                if str.lower(x) == str.lower(g[z]):
                  #every time the word is found increment the counter by 1
                    counter = counter + 1 
                  # if not found end for loob
                    break

        list1.append([str.lower(x),counter]) # store every word 
    return list1 ## return the word & the number of times it is found


  ## Function for Positional Index
def positinalIndex(sortedWord,sameWordsResult,wordslist):
    list1 = []
      ## counter for ____
    counter1 = 0
      ## loob over sortedwords "x = word"
    for x in sortedWord:
        list2 = []
          ## counter for _____
        position = 0 
          ## loob over words in the file "raw"
        for y in wordslist:
            list3 = []
            list4 = []
            position = position + 1
              ## loob over wordlist lenth & j equal to an incremented number starts with 0 to loob over every word in y "y = raw"
            for j in range(0,len(y)):
                ## x is the word in sorted word && y[j] is the word in the file
                if x == y[j]: 
                    list4.append(j+1) ## add the position of the word in file in list4
            if list4: ## true
                list3.append(("doc {}".format(position),list4))## doc {} for the document and between the {} ,it will add the POSITION 
            if list3:
                list2.append(list3)
        list1.append((x,sameWordsResult[counter1][1],list2))
        counter1+=1
    return list1


def matrix(tf,idf,sortterms):
    listt1 = []
    listt2 = []
    listt3 = [] 
    listt4 = []
    listt5 = []
    listt6 = []
    listt7 = []
    listt8 = []
    listt9 = []
    listt10 = []
    alllist=[listt1,listt2,listt3,listt4,listt5,listt6,listt7,listt8,listt9,listt10]

    l=0
    for li in range(0,len(alllist)):
      listt = []
      n = 0
      for x in tf[0]:
         listt.append(idf[n][2]*x[1][l][0][1])
         n=n+1
      alllist[li].append(listt)
      l=l+1
    data = {
        "doc1": listt1[0],
        "doc2": listt2[0],
        "doc3": listt3[0],
        "doc4": listt4[0],
        "doc5": listt5[0],
        "doc6": listt6[0],
        "doc7": listt7[0],
        "doc8": listt8[0],
        "doc9": listt9[0],
        "doc10": listt10[0]
    }
    
    df = pd.DataFrame(data, index=sortterms)

    return df

  ## Function to search about the word
def return_query_doc(query,wordslists):
    y=[]
    query_list = Tokens(query)
    lis=[]
    docs=[]
    if query_list:
        for x in wordslists:
            if all(z in x[0] for z in query_list):
                lis.append(x[1])
                docs.append(x[0])
        return lis,docs,query_list
    else:
        return y

def len_doc(tf_idf):
    turn_arr = tf_idf.to_numpy()##covert the dataframe to array
    list1 = []
    list2 = []
    count = 0
    for p in range(10):
        for i in turn_arr:
            count+=1
            list1.append(i[p])
        list2.append(round(norm((list1)),3))
        list1.clear()
    return list2       
def getWTF(x):
  try: 
    return math.log10(x)+1
  except: 
    return 0 
def queryTF(query_list,normalization,idf):
    query1 = pd.DataFrame( normalization.index)
    query1['tf'] = [1 if x in query_list else 0 for x in list(normalization.index)] ##tf
    query1['w_tf'] = query1['tf'].apply(lambda x : getWTF(x))
    query1['idf'] = idf['idf'] * query1['w_tf']
    query1['tf_idf'] = query1['w_tf'] * query1['idf']
    return query1    
    

#C:\Users\User\Desktop\IRProject\Ir.py

if __name__ == '__main__':
  ## open & read files and send the info to tokenization Function
    file1 = webtext.raw(r"C:\Users\User\Desktop\IRProject\1.txt")
    word1=Tokens(file1)
    file2 = webtext.raw(r"C:\Users\User\Desktop\IRProject\2.txt")
    word2=Tokens(file2)
    file3 = webtext.raw(r"C:\Users\User\Desktop\IRProject\3.txt")
    word3=Tokens(file3)
    file4 = webtext.raw(r"C:\Users\User\Desktop\IRProject\4.txt")
    word4=Tokens(file4)
    file5 = webtext.raw(r"C:\Users\User\Desktop\IRProject\5.txt")
    word5=Tokens(file5)
    file6 = webtext.raw(r"C:\Users\User\Desktop\IRProject\6.txt")
    word6 = Tokens(file6)
    file7 = webtext.raw(r"C:\Users\User\Desktop\IRProject\7.txt")
    word7 = Tokens(file7)
    file8 = webtext.raw(r"C:\Users\User\Desktop\IRProject\8.txt")
    word8 = Tokens(file8)
    file9 = webtext.raw(r"C:\Users\User\Desktop\IRProject\9.txt")
    word9 = Tokens(file9)
    file10 = webtext.raw(r"C:\Users\User\Desktop\IRProject\10.txt")
    word10 = Tokens(file10)


    wordslist = [word1, word2, word3, word4, word5, word6,word7,word8,word9,word10]  ## wordslists : a list of word with its filename 
    wordslists = [(word1,"file1"), (word2,"file2"), (word3,"file3"), (word4,"file4"), (word5,"file5"), (word6,"file6"),(word7,"file7"),(word8,"file8"),(word9,"file9"),(word10,"file10")]
    print("the files words: \n ");print(word1); print(word2); print(word3); print(word4); print(word5);print(word6);print(word7);print(word8);print(word9);print(word10);print("\n")
  
    wordJoin = list(set().union(word1,word2,word3,word4,word5,word6,word7,word8,word9,word10))
    sortedWord = sorted(wordJoin)
    print(f"the words after tokenization & union : \n {sortedWord} \n")

    sameWordsResult = countWord(sortedWord,wordslist)
    print(f"WORD COUNTER:\n {sameWordsResult} \n")

    positinal_indes=positinalIndex(sortedWord,sameWordsResult,wordslist)
    print(f"POSITIONAL INDEX: \n {positinal_indes} \n")
    


    tf=TF(sortedWord ,wordslist)
    print(f"The  Term Freq: \n{tf} \n")
    idf=Calc_IDF(wordslist,positinal_indes)
    dfidf = pd.DataFrame (idf, columns = ['words', 'freq','idf'])
    print(f"IDF:\n {idf} \n")
    tf_idf = matrix(tf, idf,sortedWord)
    print(f"TF-IDF: \n {tf_idf} \n")

    length_doc=len_doc(tf_idf)
    print(f"the docs length: \n {length_doc}\n")
    normlization = round(tf_idf/length_doc,3) ## كل idf for the word in one doc للكلمه هيقسمه ع length for the word in all docs  ## than it will return the normalization for each word 
    print(f"The normalization: \n {normlization} \n")
        ## take a word from tthe user & apply tokenization on that word
    query=input("enter query: ")
    matched_doc,matched_word,qu=return_query_doc(query,wordslists)
    query_list1 = queryTF(qu , normlization,dfidf)
    print(query_list1)
    print(f"The matched docs: \n {matched_doc}") 
    allFiles=[]
    for i in wordslists:
        allFiles.append(i[1])
    #print(allFiles)
    fileNum=[]
    c=0
    for i in matched_doc:
        c=0
        for n in allFiles:
            if i==n:
                fileNum.append(c)
            c+=1
    #print(fileNum)
    z=set()
    for i in matched_word:
        for b in i:
            z.add(b)
    uniqueWords = sorted(list(z))
    #print(uniqueWords)
    posInSortedWord=[]
    for i in uniqueWords:
        q=0
        for w in sortedWord :
            if i==w:
                posInSortedWord.append(q)
            q+=1
    print(posInSortedWord)
    #print(sortedWord)
    #print(m)
    e=normlization.iloc[posInSortedWord,fileNum]
    print(e)
    setAll0 = np.zeros((len(posInSortedWord),len("v")))
    df =pd.DataFrame(setAll0,index=uniqueWords)
    #print(df)
    getidf=[]
    for i in qu:
        for n in idf:
           if i==n[0]:
                getidf.append(n[2])
    length=norm(getidf) ## length for the 
    print(f"qu length: {length}")
    normaliiz=getidf/length
    print(f"The normalization: {normaliiz}")
    j=0
    for k in qu:
        count=0
        for i in uniqueWords:
            if i==k:
               df.iloc[count,:]=normaliiz[j]
            count+=1           
        j+=1
    print(df) 
    #print("\n")
    lisst=[]## normalize
    lisst=df.iloc[:,0]
    #print(lisst)
    ll=[]
    for i in range(len(e.columns)):
        xx=e.iloc[:,i]
        xx=xx.to_numpy()
        ll.append(xx)# array of  the first matrix
    #print(ll)
    cc=[]
    for i in range(len(ll)):
        vv=np.dot(lisst,ll[i])
        cc.append(vv) 
    bb={}   
    for i in range(len(cc)):
        bb[matched_doc[i]]=cc[i]
    #print(f"cosine: {bb} \n")
    sorr=sorted(bb.items(),key=lambda x:x[1],reverse=True)
    print(f"{sorr}\n")
    op=[]
    for i in sorr:
        op.append(i[0])
    #print("\n")
    print(f"matched files:{op}")