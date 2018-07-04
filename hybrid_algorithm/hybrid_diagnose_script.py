"""
hybrid diagnose script
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import time
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ddd.utilities import organise_tensor_data
from mbd.utilities import hypothesis_test
from mbd.utilities import residuals_pca
from mbd.utilities import get_conflicts_consistencies
from hybrid_algorithm.utilities import priori_vec2tup
from hybrid_algorithm.hybrid_stats import hybrid_stats
from hybrid_algorithm.hybrid_ann_diagnoser import hybrid_ann_diagnoser
from hybrid_algorithm.hybrid_ann_consistency_diagnoser import hybrid_ann_consistency_diagnoser

#data amount
small_data      = True
#settings
snr             = 20
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH        = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\") + str(snr) + "db\\"
cnn             = "cnn.pkl"
igcnn           = "igcnn.pkl"
igscnn          = "igscnn.pkl"
higscnn         = "higscnn.pkl"
higsecnn        = "higsecnn.pkl"
step_len        = 100
batch           = 5000
alpha           = 0.95

#load fe and iso
cnn             = torch.load(ANN_PATH + cnn)
cnn.eval()
igcnn           = torch.load(ANN_PATH + igcnn)
igcnn.eval()
igscnn          = torch.load(ANN_PATH + igscnn)
igscnn.eval()
higscnn         = torch.load(ANN_PATH + higscnn)
higscnn.eval()
higsecnn        = torch.load(ANN_PATH + higsecnn)
higsecnn.eval()
#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=snr, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=0, two_fault=10)
cnn_inputs = inputs.view(-1,1,5,100)
sen_res    = organise_tensor_data(inputs, res)
#priori by data
priori0  = cnn(cnn_inputs).detach().numpy()
priori1  = igcnn(inputs).detach().numpy()
priori2  = igscnn(inputs).detach().numpy()
priori3  = higscnn(sen_res).detach().numpy()
priori4  = higsecnn(sen_res).detach().numpy()

#res
conflict_table  = [[1, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]]
residuals       = residuals_pca(inputs, res)
#[0.00629772,  0.00317557,  0.67135618]
var             = np.array([0.00629772,  0.00317557,  0.67135618])

#labels
labels = labels.detach().numpy()

#stats
statistic = hybrid_stats()

#hybrid diagnosers
dia_cnn0       = hybrid_ann_diagnoser()
dia_cnn1       = hybrid_ann_consistency_diagnoser(alpha)

dia_igcnn0     = hybrid_ann_diagnoser()
dia_igcnn1     = hybrid_ann_consistency_diagnoser(alpha)

dia_igscnn0    = hybrid_ann_diagnoser()
dia_igscnn1    = hybrid_ann_consistency_diagnoser(alpha)

dia_higscnn0   = hybrid_ann_diagnoser()
dia_higscnn1   = hybrid_ann_consistency_diagnoser(alpha)

dia_higsecnn0  = hybrid_ann_diagnoser()
dia_higsecnn1  = hybrid_ann_consistency_diagnoser(alpha)

#set res set
dia_cnn1.set_res_set(conflict_table)
dia_igcnn1.set_res_set(conflict_table)
dia_igscnn1.set_res_set(conflict_table)
dia_higscnn1.set_res_set(conflict_table)
dia_higsecnn1.set_res_set(conflict_table)

#set order
order = (0,1,2,3,4,5)
dia_cnn0.set_order(order)
dia_cnn1.set_order(order)
dia_igcnn0.set_order(order)
dia_igcnn1.set_order(order)
dia_igscnn0.set_order(order)
dia_igscnn1.set_order(order)
dia_higscnn0.set_order(order)
dia_higscnn1.set_order(order)
dia_higsecnn0.set_order(order)
dia_higsecnn1.set_order(order)

statistic.add_diagnoser("cnn0")
statistic.add_diagnoser("igcnn0")
statistic.add_diagnoser("igscnn0")
statistic.add_diagnoser("higscnn0")
statistic.add_diagnoser("higsecnn0")
statistic.add_diagnoser("cnn1")
statistic.add_diagnoser("igcnn1")
statistic.add_diagnoser("igscnn1")
statistic.add_diagnoser("higscnn1")
statistic.add_diagnoser("higsecnn1")
#diagnosis number
num = 1
for label, p0, p1, p2, p3, p4, res in zip(labels, priori0, priori1, priori2, priori3, priori4, residuals):
    p0    = priori_vec2tup(p0)
    p1    = priori_vec2tup(p1)
    p2    = priori_vec2tup(p2)
    p3    = priori_vec2tup(p3)
    p4    = priori_vec2tup(p4)
    #set priori probability
    dia_cnn0.set_priori(p0)
    dia_cnn1.set_priori(p0)
    dia_igcnn0.set_priori(p1)
    dia_igcnn1.set_priori(p1)
    dia_igscnn0.set_priori(p2)
    dia_igscnn1.set_priori(p2)
    dia_higscnn0.set_priori(p3)
    dia_higscnn1.set_priori(p3)
    dia_higsecnn0.set_priori(p4)
    dia_higsecnn1.set_priori(p4)

    #residuals
    results                  = hypothesis_test(res, var, alpha)
    #add conflicts and consistencies
    dia_cnn1.set_res_values(results)
    dia_igcnn1.set_res_values(results)
    dia_igscnn1.set_res_values(results)
    dia_higscnn1.set_res_values(results)
    dia_higsecnn1.set_res_values(results)

    re_cnn0            = dia_cnn0.search(num)
    re_cnn1            = dia_cnn1.search(num)
    re_igcnn0          = dia_igcnn0.search(num)
    re_igcnn1          = dia_igcnn1.search(num)
    re_igscnn0         = dia_igscnn0.search(num)
    re_igscnn1         = dia_igscnn1.search(num)
    re_higscnn0        = dia_higscnn0.search(num)
    re_higscnn1        = dia_higscnn1.search(num)
    re_higsecnn0       = dia_higsecnn0.search(num)
    re_higsecnn1       = dia_higsecnn1.search(num)


    statistic.append_label(label)
    statistic.append_predicted("cnn0",      re_cnn0)
    statistic.append_predicted("cnn1",      re_cnn1)
    statistic.append_predicted("igcnn0",    re_igcnn0)
    statistic.append_predicted("igcnn1",    re_igcnn1)
    statistic.append_predicted("igscnn0",   re_igscnn0)
    statistic.append_predicted("igscnn1",   re_igscnn1)
    statistic.append_predicted("higscnn0",  re_higscnn0)
    statistic.append_predicted("higscnn1",  re_higscnn1)
    statistic.append_predicted("higsecnn0", re_higsecnn0)
    statistic.append_predicted("higsecnn1", re_higsecnn1)

statistic.print_stats()
print("DONE")
