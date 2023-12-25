import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
from nltk.stem import PorterStemmer
# import nltk
# nltk.download('punkt')

from collections import OrderedDict

import nltk
from nltk.stem import PorterStemmer

##################################################tokenization
def tokenization(fi):
    # stop_words= stopwords.words('english')
    # stop_words.remove('in')
    # stop_words.remove('to')
    # stop_words.remove('where')

    files_name = natsorted(os.listdir(fi))

    # print(files_name)

    document_of_terms=[]
    for files in files_name:
        
        with open(f'files/{files}', 'r')as f:
            document =f.read()
            # print(document)
        tokenized_documents = word_tokenize(document)
        terms=[]
        for word in tokenized_documents:
            # if word not in stop_words: 
            terms.append(word)
        document_of_terms.append(terms)

    print(document_of_terms)
    return document_of_terms

##################################################################stemming
def doc_stemmer(document_of_terms):
    # Instantiate the Porter Stemmer
    porter = PorterStemmer()

    # Apply stemming to each token in the documents
    stemmed_documents = [[porter.stem(token) for token in document] for document in document_of_terms]

    print(stemmed_documents)
    # Print the stemmed documents
    for i, doc in enumerate(stemmed_documents):
        print(f"Document {i + 1}: {doc}")
    return stemmed_documents

######################################################################positional_index

# document_number=0
# positional_index={}

# for document in stemmed_documents:
#     for positional , term in enumerate(document):

#         if term in positional_index:
#             positional_index[term][0]=positional_index[term][0]+1

#             if document_number in positional_index[term][1]:
#                 positional_index[term][1][document_number].append(positional)

#             else:
#                 positional_index[term][1][document_number]=[positional]


#         else:
#             positional_index[term]=[]

#             positional_index[term].append(1)

#             positional_index[term].append({})

#             positional_index[term][1][document_number]=[positional]

#     document_number+=1

# print(positional_index)

########################################################################################positional sorted
# from collections import OrderedDict
def positional_index(stemmed_documents):
    document_number = 0
    positional_index = OrderedDict()

    for document in stemmed_documents:
        for positional, term in enumerate(document):

            if term in positional_index:
                positional_index[term][0] = positional_index[term][0] + 1

                if document_number in positional_index[term][1]:
                    positional_index[term][1][document_number].append(positional)

                else:
                    positional_index[term][1][document_number] = [positional]

            else:
                positional_index[term] = []

                positional_index[term].append(1)

                positional_index[term].append(OrderedDict())

                positional_index[term][1][document_number] = [positional]

        document_number += 1

    # Sort the positional index
    sorted_positional_index = OrderedDict(sorted(positional_index.items()))

    print(sorted_positional_index)

    return sorted_positional_index



#########################################################################phrase query

# query='fools fear'
def doc_phrase_search(query , sorted_positional_index ):
    porter = PorterStemmer()
    query_tokens = word_tokenize(query)
    stemmed_query = [porter.stem(token) for token in query_tokens]
    query= ' '.join(stemmed_query)
    #make list of ten empty lists
    final_list = [[]for i in range (10)]
    print(final_list)
    porter = PorterStemmer()
    for word in query.split():
        stemmed_token = porter.stem(word)
        for key in sorted_positional_index[stemmed_token][1].keys():

            if final_list[key-1] !=[]:
                if final_list[key-1][-1]==sorted_positional_index[stemmed_token][1][key][0]-1:
                    final_list[key-1].append(sorted_positional_index[stemmed_token][1][key][0])



            else:
                final_list[key-1].append(sorted_positional_index[stemmed_token][1][key][0])

    print(final_list)

    for position , list in enumerate(final_list, start=1):
        # print(position , list)
        if len(list) ==len(query.split()):
            print(position)


