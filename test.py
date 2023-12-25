import pandas as pd
import math
import numpy as np
from phase1 import *

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def get_docs_length(col, term_freq_inverse_doc_freq):
        return np.sqrt(term_freq_inverse_doc_freq[col].apply(lambda x : x**2).sum())

def get_term_freq(doc , all_words):
        words_found=dict.fromkeys(all_words,0)
        for word in doc:
            words_found[word] +=1
        return words_found

def get_weighted_term_freq(x):
        if x>0:
            return math.log(x)+1
        return 0

def get_normalized(col , x , document_length):
        try:
            return x / document_length[col+'_len'].values[0]
        
        except:
            return 0

def tf_IDF_norm(stemmed_documents):
    documents= stemmed_documents
    all_words=[]

    for doc in documents:
        for word in doc:
            all_words.append(word)

    # print(dict.fromkeys(all_words,0))

    # def get_term_freq(doc):
    #     words_found=dict.fromkeys(all_words,0)
    #     for word in doc:
    #         words_found[word] +=1
    #     return words_found

    term_freq= pd.DataFrame(get_term_freq(documents[0], all_words).values(),index=get_term_freq(documents[0], all_words).keys())


    for i in range(1,len(documents)):
        term_freq[i]= get_term_freq(documents[i] , all_words).values()

    # print(term_freq)

    term_freq.columns=['doc'+ str(i) for i in range(1,11)]
    print(term_freq)


    # def get_weighted_term_freq(x):
    #     if x>0:
    #         return math.log(x)+1
    #     return 0

    for i in range (1,len(documents)+1):
        term_freq['doc'+str(i)]=term_freq['doc'+str(i)].apply(get_weighted_term_freq)

    print(term_freq)
    # idf is log(doc num/ term freq in doc)
    tfd=pd.DataFrame(columns=['freq','idf'])

    for i in range(len(term_freq)):

        frequency=term_freq.iloc[i].values.sum()

        tfd.loc[i ,'freq']= frequency

        tfd.loc[i ,'idf']=math.log10(10 / float((frequency)) )

    tfd.index= term_freq.index
    print(tfd)    


    ##### TF IDF #####

    term_freq_inverse_doc_freq= term_freq.multiply(tfd['idf'], axis=0)

    print(term_freq_inverse_doc_freq)


    ####### doc length

    document_length=pd.DataFrame()

    # def get_docs_length(col, term_freq_inverse_doc_freq):
    #     return np.sqrt(term_freq_inverse_doc_freq[col].apply(lambda x : x**2).sum())

    for column in term_freq_inverse_doc_freq.columns:
        document_length.loc[0 , column+'_len']= get_docs_length(column , term_freq_inverse_doc_freq)

    print(document_length)


    #### normalized TF IDF ######


    normalized_term_freq_idf= pd.DataFrame()

    for column in term_freq_inverse_doc_freq.columns:
        normalized_term_freq_idf[column]= term_freq_inverse_doc_freq[column].apply(lambda x : get_normalized(column , x , document_length))
     

    print(normalized_term_freq_idf)

    return normalized_term_freq_idf, tfd



def phrse_query(q,normalized_term_freq_idf,tfd):
    # def query_norm_idf(q):
    # phrse_query('fools fear')
    porter = PorterStemmer()
    query_tokens = word_tokenize(q)
    stemmed_query = [porter.stem(token) for token in query_tokens]
    q= ' '.join(stemmed_query)
    query= pd.DataFrame(index=normalized_term_freq_idf.index)
    query['tf']=[1 if x in q.split() else 0 for x in normalized_term_freq_idf.index]
    query['w_tf']= query['tf'].apply(lambda x: get_weighted_term_freq(x))
    product = normalized_term_freq_idf.multiply(query['w_tf'], axis= 0)
                # product
    query['idf'] = tfd['idf'] * query['w_tf']
    query['tf_idf'] = query['idf'] * query['w_tf']
    query['norm']=0
    for i in range(len(query)):
        query['norm'].iloc[i]= float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
    product2=product.multiply(query['norm'], axis=0)
        # product2
    print(query)
    query_length=math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]]))
    print("query_length")
    print(query_length)
    scores = {}
    for col in product2.columns:
        if 0 in product2[col].loc[q.split()].values:
            pass
        else:
            scores[col]=product2[col].sum()

    print("sum")
    print(scores)
    ############ doc terms scores
    prod_res= product2[scores.keys()].loc[q.split()]
    print(prod_res)
    ###########similarity
    prod_sum=prod_res.sum()
    final_score=sorted(prod_sum.items(), key= lambda x :x[1], reverse= True)
    print(final_score)
    for doc in final_score:
        print(doc[0], end=' ')
    return query, query_length, scores

from sklearn.metrics.pairwise import cosine_similarity

def bool( positional_index, normalized_term_freq_idf, tfd):
    #  AND , OR , NOT اشتغل 


    # Instantiate Porter Stemmer
    porter = PorterStemmer()

    def preprocess_query(query):
        # Tokenize the query and stem each term
        token_docs = word_tokenize(query)
        prepared_doc = [porter.stem(term) for term in token_docs]
        return prepared_doc

    

    def calculate_similarity(query_vector, document_vectors):
        # Assuming query_vector and document_vectors are NumPy arrays
        similarity_scores = cosine_similarity(query_vector.reshape(1, -1), document_vectors)
        return similarity_scores[0]

    def put_query(query, positional_index, normalized_term_freq_idf, tfd):
        boolean_operators = ['AND', 'OR', 'NOT']
        boolean_operator = None

        # Check if any boolean operator is present in the query
        for operator in boolean_operators:
            if operator in query:
                boolean_operator = operator
                break

        # If no boolean operator is present, treat the entire query as a phrase
        if boolean_operator is None:
            preprocessed_query = preprocess_query(query)
            positions = find_positions(preprocessed_query, positional_index)
        else:
            # Split the query based on the boolean operator
            query_parts = query.split(boolean_operator)
            preprocessed_query1 = preprocess_query(query_parts[0].strip())
            preprocessed_query2 = preprocess_query(query_parts[1].strip())
            # query=query.replace(boolean_operator, '')

            if boolean_operator == 'AND':
                positions = perform_and_operation(preprocessed_query1, preprocessed_query2, positional_index)
            elif boolean_operator == 'OR':
                positions = perform_or_operation(preprocessed_query1, preprocessed_query2, positional_index)
            elif boolean_operator == 'NOT':
                positions = perform_not_operation(preprocessed_query1, preprocessed_query2, positional_index)

        return positions

    def find_positions(preprocessed_query, positional_index):
        lis = [[] for i in range(10)]  
        for term in preprocessed_query:
            if term in positional_index.keys():
                for key in positional_index[term][1].keys():
                    if lis[key] != []:
                        if lis[key][-1] == positional_index[term][1][key][0] - 1:
                            lis[key].append(positional_index[term][1][key][0])
                    else:
                        lis[key].append(positional_index[term][1][key][0])

        positions = []
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(preprocessed_query):
                positions.append('document ' + str(pos))
        return positions

    def perform_and_operation(query1, query2, positional_index):
        positions1 = find_positions(query1, positional_index)
        positions2 = find_positions(query2, positional_index)
        # q= 'antoni brutu'
        # boolean_operators = ['AND', 'OR', 'NOT']
        # print(user_query)
        # q=user_query
        # ####
        # qu= pd.DataFrame(index=normalized_term_freq_idf.index)
        # qu['tf']=[1 if x in q.split() else 0 for x in normalized_term_freq_idf.index]
        # qu['w_tf']= qu['tf'].apply(lambda x: get_weighted_term_freq(x))
        # product = normalized_term_freq_idf.multiply(qu['w_tf'], axis= 0)
        #             # product
        # qu['idf'] = tfd['idf'] * qu['w_tf']
        # qu['tf_idf'] = qu['idf'] * qu['w_tf']
        # qu['norm']=0
        # for i in range(len(qu)):
        #     qu['norm'].iloc[i]= qu['idf'].iloc[i] / math.sqrt(sum(qu['idf'].values**2))
        # product2=product.multiply(qu['norm'], axis=0)
        #     # product2
        # print(qu)
        # ####
        return list(set(positions1) & set(positions2))

    def perform_or_operation(query1, query2, positional_index):
        positions1 = find_positions(query1, positional_index)
        positions2 = find_positions(query2, positional_index)
        return list(set(positions1) | set(positions2))

    def perform_not_operation(query1, query2, positional_index):
        positions1 = find_positions(query1, positional_index)
        positions2 = find_positions(query2, positional_index)
        return list(set(positions1) - set(positions2))


    user_query = input("Enter your boolean query: ")
    preprocessed_query = preprocess_query(user_query)
    print('Original Query:', user_query)
    print('Preprocessed Query:', preprocessed_query)
    result = put_query(user_query, positional_index, normalized_term_freq_idf, tfd)
    print('Matched Documents:', result)
    query_vector = result['tf_idf'].values  # Adjust this line based on your data structure

    # Assuming document_vectors is a DataFrame containing tf_idf values for all documents
    document_vectors = normalized_term_freq_idf.T.values  # Adjust this line based on your data structure

    similarity_scores = calculate_similarity(query_vector, document_vectors)
    print('Similarity Scores:', similarity_scores)