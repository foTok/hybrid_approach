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
from mbd.utilities import get_conflicts
from hybrid_algorithm.utilities import priori_vec2tup
from hybrid_algorithm.hybrid_stats import hybrid_stats
from hybrid_algorithm.hybrid_ann_diagnoser import hybrid_ann_diagnoser
from hybrid_algorithm.hybrid_ann_consistency_diagnoser import hybrid_ann_consistency_diagnoser

#data amount
small_data      = True
#settings
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH        = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH2       = PATH + "\\mbd\\ann_model\\"
dia_file        = "DIA.pkl"
hdia_file       = "HDIA.pkl"
bsshdia_file    = "BSSHDIA.pkl"
pdia_file       = "PDIA.pkl"
step_len        = 100
batch           = 5000
p0              = 0.95

#load fe and iso
DIA         = torch.load(ANN_PATH + dia_file)
DIA.eval()
HDIA        = torch.load(ANN_PATH + hdia_file)
HDIA.eval()
BSSHDIA     = torch.load(ANN_PATH + bsshdia_file)
BSSHDIA.eval()
PHDIA       = torch.load(ANN_PATH2 + pdia_file)
PHDIA.eval()
#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=1)
#residual 0
r0      = np.array([r[0] for r in res])
r0      = torch.Tensor(r0)
r0      = r0.view(-1, 1, 100)
#priori by data
ann_start       = time.clock()
priori_by_data  = DIA(inputs).detach().numpy()
ann_cost        = time.clock() - ann_start
#priori by hybrid
sen_res          = organise_tensor_data(inputs, res)
hann_start       = time.clock()
priori_by_hybrid = HDIA(sen_res).detach().numpy()
hann_cost        = time.clock() - hann_start
#priori by bbshybrid
bsshann_start       = time.clock()
priori_by_bsshybrid = BSSHDIA(sen_res).detach().numpy()
bsshann_cost        = time.clock() - bsshann_start
#priori by pdia
hannm_start         = time.clock()
priori_by_pdia      = PHDIA(r0).detach().numpy()
hannm_cost          = time.clock() - hannm_start + bsshann_cost
#priori by ddd+mbd
priori_by_dm        = priori_by_bsshybrid.copy()
priori_by_dm[:, 0]  = priori_by_dm[:, 0] * 0.2 + priori_by_pdia[:, 0] * 0.8
priori_by_dm[:, 1]  = priori_by_dm[:, 1] * 0.2 + priori_by_pdia[:, 1] * 0.8
priori_by_dm[:, 5]  = priori_by_dm[:, 5] * 0.2 + priori_by_pdia[:, 2] * 0.8
#res
conflict_table  = [[1, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]]
residuals       = residuals_pca(inputs, res)
var             = np.array([0.00629772,  0.00317557,  0.67135618])

#labels
labels = labels.detach().numpy()

#stats
statistic = hybrid_stats()

#hybrid diagnosers
#ANN
ann0    = hybrid_ann_diagnoser()
ann     = hybrid_ann_consistency_diagnoser(p0)
#Hybrid ANN
hann0   = hybrid_ann_diagnoser()
hann    = hybrid_ann_consistency_diagnoser(p0)
#BSSHybrid
bsshann0= hybrid_ann_diagnoser()
bsshann = hybrid_ann_consistency_diagnoser(p0)
#ANN + MBD
hannm0  = hybrid_ann_diagnoser()
hannm   = hybrid_ann_consistency_diagnoser(p0)
#set order
order = (0,1,2,3,4,5)
ann.set_order(order)
hann.set_order(order)
bsshann.set_order(order)
hannm.set_order(order)
ann0.set_order(order)
hann0.set_order(order)
bsshann0.set_order(order)
hannm0.set_order(order)

statistic.add_diagnoser("ann")
statistic.add_diagnoser("hann")
statistic.add_diagnoser("bsshann")
statistic.add_diagnoser("hannm")
statistic.add_diagnoser("ann0")
statistic.add_diagnoser("hann0")
statistic.add_diagnoser("bsshann0")
statistic.add_diagnoser("hannm0")
#diagnosis number
num = 1
for label, d_priori, h_priori, bh_priori, hm_priori, res in zip(labels, priori_by_data, priori_by_hybrid, priori_by_bsshybrid, priori_by_dm, residuals):
    priori_d    = priori_vec2tup(d_priori)
    priori_h    = priori_vec2tup(h_priori)
    priori_bh   = priori_vec2tup(bh_priori)
    priori_hm   = priori_vec2tup(hm_priori)

    #set priori probability
    ann.set_priori(priori_d)
    hann.set_priori(priori_h)
    bsshann.set_priori(priori_bh)
    hannm.set_priori(priori_hm)
    ann0.set_priori(priori_d)
    hann0.set_priori(priori_h)
    bsshann0.set_priori(priori_bh)
    hannm0.set_priori(priori_hm)

    #residuals
    results         = hypothesis_test(res, var, p0)
    conflicts       = get_conflicts(results, conflict_table)

    #add conflicts
    ann.add_conflicts(conflicts)
    hann.add_conflicts(conflicts)
    bsshann.add_conflicts(conflicts)
    hannm.add_conflicts(conflicts)

    dia_ann      = ann.search(num)
    dia_hann     = hann.search(num)
    dia_bsshann  = bsshann.search(num)
    dia_hannm    = hannm.search(num)
    dia_ann0     = ann0.search(num)
    dia_hann0    = hann0.search(num)
    dia_bsshann0 = bsshann0.search(num)
    dia_hannm0   = hannm0.search(num)


    statistic.append_label(label)
    statistic.append_predicted("ann",       dia_ann)
    statistic.append_predicted("hann",      dia_hann)
    statistic.append_predicted("bsshann",   dia_bsshann)
    statistic.append_predicted("hannm",     dia_hannm)
    statistic.append_predicted("ann0",      dia_ann0)
    statistic.append_predicted("hann0",     dia_hann0)
    statistic.append_predicted("bsshann0",  dia_bsshann0)
    statistic.append_predicted("hannm0",    dia_hannm0)

statistic.print_stats()
#ann time cost
print("ann      cost=",       ann_cost)
print("hann     cost=",       hann_cost)
print("bsshann  cost=",       bsshann_cost)
print("hannm    cost=",       hannm_cost)
#ann time search cost
print("ann      time cost=",  ann.time_cost())
print("hann     time cost=",  hann.time_cost())
print("bsshann  time cost=",  bsshann.time_cost())
print("hannm    time cost=",  hannm.time_cost())
print("ann      time cost0=", ann0.time_cost())
print("hann     time cost0=", hann0.time_cost())
print("bsshann  time cost0=", bsshann0.time_cost())
print("hannm0   time cost=",  hannm0.time_cost())

print("DONE")
