from phase1 import *
# from test import *
from phrase_query import *


doc_terms=tokenization('files')

stemmed_doc= doc_stemmer(doc_terms)

sorted_pos_index=positional_index(stemmed_doc)

doc_phrase_search('fools fear' , sorted_pos_index)

tf_idf_norm_res, tfd_res = tf_IDF_norm(stemmed_doc)

q=input("please enter the query : ")
phrse_query(q, tf_idf_norm_res, tfd_res)
bool(sorted_pos_index , q)


# bool(sorted_pos_index ,tf_idf_norm_res, tfd_res)
