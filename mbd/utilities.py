"""
some utilities
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import random
import numpy as np
from math import sqrt
from bpsk_navigate.bpsk_generator import Bpsk
from bpsk_navigate.utilities import compose_file_name
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from scipy import stats

def residuals_pca(inputs, res):
    """
    residuals of pesude code, carriar and amplifier
    """
    length              = len(res)
    batch_data          = np.zeros((length, 3))
    res                 = np.array(res)
    #res12
    res                 = np.mean(np.abs(res), axis=2)
    batch_data[:, :2]   = res[:, :2]
    #res3
    inputs              = inputs.detach().numpy()
    s3                  = inputs[:, 3]
    s4                  = inputs[:, 4]
    batch_data[:, -1]   = np.mean(np.abs( s4 - 10 * s3), axis=1)
    return batch_data


def hypothesis_test(res, var, alpha):
    """
    return the conflicts based on res, var and confidence
    """
    mean        = np.zeros((1, len(res)))
    interval    = stats.norm.interval(alpha, mean, np.sqrt(var))
    result      = []
    UP          = interval[1].reshape(3)
    DOWN        = interval[0].reshape(3)
    for r, up, down in zip(res, UP, DOWN):
        if r < up and r > down:
            result.append(True)
        else:
            result.append(False)
    return result

def get_conflicts(results, conflict_table):
    """
    find conflict based on results and conflic table
    """
    conflicts       = []
    for c, pas in zip(conflict_table, results):
        if not pas:
            conflicts.append(c)
    return conflicts

def sample_parameters(N, fault_type, grids, begin, end, pref, para_set):
    """
    sample parameters
    """
    parameters = {}
    #single-fault
    if pref == 1 or pref == 3:
        for fault, grid, be, en in zip(fault_type, grids, begin, end):
            para = sample_para1(N, fault, grid, be, en, para_set)
            parameters[fault] = para
    if pref == 2 or pref == 3:
        for i in range((len(fault_type))):
            for j in range(i+1, len(fault_type)):
                fault = (fault_type[i], fault_type[j])
                grid  = (grids[i], grids[j])
                be    = (begin[i], begin[j])
                en    = (end[i], end[j])
                para  = sample_para2(N, fault, grid, be, en, para_set)
                parameters[fault] = para
    return parameters

def sample_para0(fault, begin, end):
    if not (isinstance(begin, tuple) or isinstance(begin, list)):
        while True:
            para    = random.uniform(begin, end)
            if abs(para) > 0.01:
                break
    else:#TMB fault
        begin1  = begin[0]
        begin2  = begin[1]
        end1    = end[0]
        end2    = end[1]
        sigma   = random.uniform(begin1, end1)
        f_d     = random.uniform(begin2, end2)
        para    = (sigma, f_d)
    return para

def sample_para1(N, fault, grid, begin, end, para_set):
    """
    sample parameters for a single-fault
    N: int
    fault_ytpe: str
    """
    if fault not in para_set:
        para_set[fault] = []
    para_list = []
    max_iter  = 10*N
    it        = 0 #iter count
    n         = 0 #count
    while n < N:
        it = it + 1
        if it > max_iter:
            break
        para = sample_para0(fault, begin, end)
        if is_new(fault, grid, para, para_set[fault]):
                para_list.append(para)
                para_set[fault].append(para)
                n = n + 1
    return para_list

def sample_para2(N, fault, grid, begin, end, para_set):
    """
    fault: tuple
    """
    if fault not in para_set:
        para_set[fault] = []
    para_list = []
    max_iter  = 10*N
    it        = 0 #iter count
    n         = 0 #count
    while n < N:
        it = it + 1
        if it > max_iter:
            break
        p1 = sample_para0(fault[0], begin[0], end[0])
        p2 = sample_para0(fault[1], begin[1], end[1])
        if is_new(fault, grid, (p1, p2), para_set[fault]):
            para_list.append((p1, p2))
            para_set[fault].append((p1, p2))
            n = n + 1
    return para_list

def is_new(fault_type, grid, para, para_set):
    """
    check if para is new for para_set
    """
    if isinstance(fault_type, str):
        for para0 in para_set:
            if distance1(para0, para) < grid:
                return False
    else:
        for para0 in para_set:
            d1, d2 = distance2(para0, para)
            if (d1<grid[0]) and (d2<grid[1]):
                return False
    return True

def distance1(para0, para1):
    """
    compute the distance between two points for single-fault
    """
    if not isinstance(para0, tuple):
        return abs(para0 - para1)
    else:
        return sqrt((para0[0] - para1[0])**2 + (para0[1] - para1[1])**2)

def distance2(para0, para1):
    """
    compute the distance betwwen two points for two-fault
    """
    return distance1(para0[0], para1[0]), distance1(para0[1], para1[1])

def sample_data(fault_time, end_time, parameters, path):
    """
    sample data based on parameters and save them in path
    """
    normal_path = path + "normal\\"
    file_list  = []
    for fault in parameters:
        if isinstance(fault, str):#single-fault
            for para in parameters[fault]:
                simulator = Bpsk()
                simulator.insert_fault_para(fault, para)
                simulator.insert_fault_time("all", fault_time)
                data = simulator.generate_signal(end_time)
                file_name = compose_file_name(fault, para)
                #add to file list
                file_list.append(file_name)
                bpsk_pre = Bpsk()
                data_pre = bpsk_pre.generate_signal_with_input(end_time, data[:, 0])
                #save data
                np.save(path + file_name, data)
                np.save(normal_path + file_name, data_pre)
        else:
            for para in parameters[fault]:
                fault1, fault2 = fault
                para1, para2   = para
                bpsk = Bpsk()
                bpsk.insert_fault_para(fault1, para1)
                bpsk.insert_fault_para(fault2, para2)
                bpsk.insert_fault_time("all", fault_time)
                data = bpsk.generate_signal(end_time)
                file_name = compose_file_name(fault, para)
                #add to file list
                file_list.append(file_name)
                bpsk_pre = Bpsk()
                data_pre = bpsk_pre.generate_signal_with_input(end_time, data[:, 0])
                #save data
                np.save(path + file_name, data)
                np.save(normal_path + file_name, data_pre)
    return file_list

def add_files(path, file_list, mana, step_len, snr):
    """
    add files in file_list into tank
    """
    for the_file in file_list:
        mana.read_data(path + the_file, step_len=step_len, snr=snr, norm=True)
