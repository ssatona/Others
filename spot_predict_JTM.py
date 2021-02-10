import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import gensim
from operator import itemgetter


with open('theta.pickle', 'rb') as f:
    theta = pickle.load(f)
with open('all_keitaiso_byDocs.pickle', 'rb') as f2:
    all_keitaiso_byDoc = pickle.load(f2)
with open('dataset_cmp.pickle', 'rb') as f2:
    dataset_cmp = pickle.load(f2)
with open('dict_inv_cmp.pickle', 'rb') as f3:
    dict_inv_cmp = pickle.load(f3)
with open('psi.pickle', 'rb') as psi_f:
    psi = pickle.load(psi_f)
with open('word_ranking_byTopic0.pickle', 'rb') as word_r:
    word_rank = pickle.load(word_r)
with open('complement_keitaiso.pickle', 'rb') as cmp_keitaiso:
    cmp_info = pickle.load(cmp_keitaiso)

dict_cmp = gensim.corpora.Dictionary.load_from_text('complement.dict')
#print(dict_cmp)
#print(theta.shape)
#print(psi.shape)
#print(len(word_rank))
#print(dataset_cmp)
#print(len(all_keitaiso_byDoc))


#---------------文書１の補助情報がsである確率(トピックモデル p.84)--------------------
comp_prob_byDoc = np.zeros((len(all_keitaiso_byDoc),len(dict_inv_cmp)))

for doc in range(len(all_keitaiso_byDoc)):

    for s in range(len(dict_inv_cmp)):
        theta_psi_list = []

        for k in range(len(psi)):
            theta_by_psi = theta[doc][k] * psi[k][s]
            theta_psi_list.append(theta_by_psi)
        
        comp_prob_byDoc[doc][s] = sum(theta_psi_list)


#-----------------補助情報確率--------------------------------------------------
#numpyをリスト化
comp_prob_byDoc_list = comp_prob_byDoc.tolist()

#補助情報リスト
cmp_value = []
for i in dict_cmp:
    #print(dict_cmp[i])
    cmp_value.append(dict_cmp[i])

#print(len(cmp_value))
#print(sum(comp_prob_byDoc_list[0]))
#print(comp_prob_byDoc_list)
#print('comp_prob_byDoc_list',len(comp_prob_byDoc_list[0]))
#print('cmp_value',len(cmp_value))

sort_list = []
sort_list_top20 = []
comp_rank = []
for row, ac_spot in zip(comp_prob_byDoc_list,cmp_info):
    l_t_name_value = zip(cmp_value, row)
    l_t_name_value_sort = sorted(l_t_name_value, reverse = True, key = itemgetter(1))
    for i, row2 in enumerate(l_t_name_value_sort):
        #print(row2[0],ac_spot)
        if row2[0] == ac_spot[0]:
            print(row2[0],ac_spot[0])
            comp_rank.append(i)
            break
    sort_list.append(l_t_name_value_sort)
    sort_list_top20.append(l_t_name_value_sort[0:20])

#----------------最大値のみの表示-------------------------------------------------
# max_list = []
# for row in comp_prob_byDoc:
#     max_index = np.argmax(row)
#     com_prob_max = row[max_index]
#     com_name_max = dict_cmp[max_index]
#     max_list.append([com_name_max,com_prob_max])

# print(max_list)
#--------------------出力-------------------------------------------------------

#comp_prob_byDoc_df = pd.DataFrame(data=max_list, columns=['pred_name', 'prob'])
comp_prob_byDoc_sorted20_df = pd.DataFrame(data=sort_list_top20)
comp_rank_df = pd.DataFrame(data=comp_rank, columns=['cmp_rank'])
cmp_info_df = pd.DataFrame(data=cmp_info, columns=['cmp_actual_name'])
prob_actual_df = pd.concat([cmp_info_df, comp_rank_df, comp_prob_byDoc_sorted20_df], axis=1)
prob_actual_df.to_csv("prob_actual_df_only_foreign.csv",index=False)

#df_Doc = pd.DataFrame(data=comp_prob_byDoc, columns=dict_cmp, dtype='float')