#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from scipy import signal
import cvxopt as cv
import cvxopt.solvers
import neurokit2 as nk
from sklearn.metrics import auc
import statistics

SEG_START = -1
SEG_END = 10

def get_scores(file_path):
    df=pd.read_excel(file_path)
    return df["分数"].iloc[:30]
 
#获取被试其中一组的时间戳（一共三组）
def get_timestamp(file_path):
    timestamp_parsed=[]
    floors=[]
    with open(file_path,'r',encoding='utf-8-sig') as file:
        counter=0
        for line in file:
            if (counter%2==0):
                timestamp_parsed.append(line.replace("/","-").strip("\n"))    
            else:
                floors.append(line.strip("\n").strip("floor:"))
            counter+=1
    # parsed_data=pd.DataFrame()
   # parsed_data["timestamps"]=timestamp_parsed
   # parsed_data["floors"]=floor
    return timestamp_parsed

def raw_data_segmentation(biopac_file,time_list):
    eda_segmented=[]
    ecg_segmented=[]
    ppg_segmented=[]
    output_timelist=[]
    df=pd.read_csv(biopac_file, sep='\t')
    start_time= datetime.datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S%f')
    df["timestamp"]=df["timestamp"].map(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%f')-start_time).total_seconds())
    for i in range(len(time_list)):
        time_list[i]=datetime.datetime.strptime(time_list[i], '%Y-%m-%d %H:%M:%S%f')
    timestamps=[]
    for i in range(len(time_list)):
        timestamps.append((time_list[i]-time_list[0]).total_seconds())
    df_list=[]
    for i in range(1,len(time_list)):
        df_list.append(df[(df["timestamp"]>=timestamps[i]+SEG_START) & (df["timestamp"]<(timestamps[i]+SEG_END))])
    for i in range(10):
        output_timelist.append(df_list[i]["timestamp"]-timestamps[i+1]) 
        eda_segmented.append(df_list[i]["channel_13"])
        ecg_segmented.append(df_list[i]["channel_8"])
        ppg_segmented.append(df_list[i]["channel_0"])   
    return output_timelist, eda_segmented, ecg_segmented, ppg_segmented

def fix_blink(list, fps=90):
    length = len(list)
    blink_list = []
    total_blink_num = 0
    total_blink_time = 0

    # fix -1 at the beginning
    k0 = 0
    if list[k0] == -1 or list[k0] == 0:
        k_start = k0
        while ((list[k0] == -1 or list[k0] == 0) and k0 + 1 < length):
            k0 += 1
        k_end = k0 - 1

        # interpolation
        for i in range(k_start, k_end + 1):
            list[i] = list[k_end + 1]


        # update features
        blink_list.append([k_start / fps, k_end / fps])
        total_blink_num += 1
        total_blink_time += (k_start - k_end) / fps

    # fix -1 at the end
    k1 = length - 1
    if list[k1] == -1 or list[k1] == 0:
        k_end = k1
        while ((list[k1] == -1 or list[k1] == 0) and k1 - 1 > 0):
            k1 -= 1
        k_start = k1 + 1

        # interpolation
        for i in range(k_start, k_end + 1):
            list[i] = list[k_start - 1]

        # update features
        blink_list.append([k_start / fps, k_end / fps])
        total_blink_num += 1
        total_blink_time += (k_start - k_end) / fps

    # fix -1 in the middle
    k = 0
    while k + 1 < length :
        k += 1
        if list[k] == -1 or list[k] == 0:
            k_start = k
            while (list[k] == -1 or list[k] == 0) and (k + 1 < length):
                k += 1
            k_end = k-1

            # interpolation (linear)
            x1, x2 = list[k_start - 1], list[k_end + 1]
            for i in range(k_start, k_end + 1):
                list[i] = x1 + (x2 - x1) * (i - k_start + 1) / (k_end - k_start + 2)

            # update features
            blink_list.append([k_start / fps, k_end / fps])
            total_blink_num += 1
            total_blink_time += (k_start - k_end) / fps

    return(list)

def load_eye_data(filename, start_time=0, end_time=5):
    df = pd.read_csv(filename, engine='python', encoding='utf-8-sig')
    df_segment = df[(df['时间戳'] >= start_time) & (df['时间戳'] < end_time)]
    pupil_l_seg = df_segment['左眼瞳孔直径']
    pupil_r_seg = df_segment['右眼瞳孔直径']
    eye_l_seg = df_segment['左眼睁眼程度']
    eye_r_seg = df_segment['右眼睁眼程度']
    return pupil_l_seg.tolist(), pupil_r_seg.tolist(), eye_l_seg.tolist(), eye_r_seg.tolist()

def eye_data_filter(raw_data, fps = 90, threshold = 5):
    # 5阶，采样频率=90Hz，目标滤波频率=5Hz，参数2=2*5/90
    b, a = signal.butter(5, 2*threshold / fps, 'lowpass')
    filtedData = signal.filtfilt(b, a, raw_data)
    return filtedData

def eda_filter(raw_eda,timestamp):
    numtaps = 5
    # 1Hz低通滤波
    cutoff = 1
    time_total = timestamp.values[-1]
    i = 5
    cutoff_normalised=2*cutoff/(len(timestamp)/time_total)
    b, a = signal.butter(i, cutoff_normalised, 'lowpass') 
    filtered_data = signal.filtfilt(b, a, raw_eda)
    filtered_data=(filtered_data-filtered_data.mean())
    return filtered_data

def eda_filter_normalize(raw_eda,timestamp):
    numtaps = 5
    # 1Hz低通滤波
    cutoff = 1
    time_total = timestamp.values[-1]
    i = 5
    cutoff_normalised=2*cutoff/(len(timestamp)/time_total)
    b, a = signal.butter(i, cutoff_normalised, 'lowpass') 
    filtered_data = signal.filtfilt(b, a, raw_eda)
    filtered_data = (filtered_data-filtered_data.mean())/filtered_data.std()
    return filtered_data

#EDA分解算法，见github cvxEDA
def cvxEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """

    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n + nC]
    tonic = B * l + C * d
    q = res['x'][:n]
    p = A * q
    phasic = M * q
    #e = eda - phasic - tonic
    phasic = np.array(phasic)[:, 0]
    tonic = np.array(tonic)[:, 0]
    #    results = (np.array(a).ravel() for a in (r, t, p, l, d, e, obj))
 
    return (tonic,phasic)



root_path=r"C:\被试"

# participants: a list to hold all participants instances
participants=[]

class Participant():
    def __init__(self):
        self.labels = None
        self.timestamps = []
        self.raw_eda = None
        self.raw_ecg = None
        self.raw_ppg = None
        self.eda_processed = None
        self.eda_normalized = None
        self.eda_normalized_phasic = None
        self.eda_phasic = None
        self.eda_tonic = None
        self.ecg_processed = None
        self.ppg_processed = None
        self.hrv = None
        self.features = None
        self.name = ""
        self.raw_eda_peaceful = None
        self.raw_ppg_peaceful = None

        # for pupil dilation data & eye open degree
        self.pupil_left = None
        self.pupil_right = None

        self.pupil_left_base = None
        self.pupil_right_base = None
        self.pupil_left_lightreflex = None
        self.pupil_right_lightreflex = None

        self.eye_left = None
        self.eye_right = None
        self.eye_timestamps = []

print("Loading data...")
t1 = time.time()

# extract data from folders, and attach them to participants instances
for item in os.listdir(root_path):
    raw_eda_ = []
    raw_ecg_ = []
    raw_ppg_ = []

    timestamps_ = []
    eye_timestamps = []
    time_list = []

    pupil_l = []
    pupil_r = []
    pupil_l_base = []
    pupil_r_base = []
    eye_l = []
    eye_r = []

    participant_files_path = root_path + "\\" + item

    # new instance
    new_participant = Participant()
    new_participant.name = item

    print("Extracting from Participant: ", item)

    # extract all timestamps
    for i in range(1, 4):
        sub_file = participant_files_path + "\\" + str(i)
        # print(sub_file+"Sub")
        if ("TimeLogger.txt" in os.listdir(sub_file)):
            time_list.append(get_timestamp(sub_file + "\\TimeLogger.txt"))
        else:
            print("missing TimeLogger file!")

    counter=0

    # load BIOPAC data, and scores data
    for file_ in os.listdir(participant_files_path):
        # print(file_)
        if (file_.find("BIOPAC_data") != -1):
            raw_data_file_path = participant_files_path + "\\" + file_
            for time_ in time_list:
                timestamps_sub, raw_eda_sub, raw_ecg_sub, raw_ppg_sub = raw_data_segmentation(raw_data_file_path, time_)
                timestamps_ += timestamps_sub
                raw_eda_ += raw_eda_sub
                raw_ecg_ += raw_ecg_sub
                raw_ppg_ += raw_ppg_sub
        elif (file_.find("分数记录.xls") != -1):
            score_file_path = participant_files_path+"\\"+file_
            new_participant.labels = get_scores(score_file_path)

    # load eye data baseline (light reflex)
    time_duration = 5
    for i in range(1, 4):
        sub_file = participant_files_path + "\\" + str(i)

        # load pupil/eye light reflex baseline
        for item in os.listdir(sub_file):
            if (item == "1.csv"):
                filename = sub_file + "\\" + item
                # print(filename)
                pupil_l_seg, pupil_r_seg, eye_l_seg, eye_r_seg = load_eye_data(filename, end_time=time_duration)

                # fix blink
                pupil_r_base_seg = eye_data_filter(fix_blink(pupil_r_seg))
                pupil_l_base_seg = eye_data_filter(fix_blink(pupil_l_seg))

                # load eye data timestamps
                eye_timelength = len(eye_l_seg)
                eye_timestamps_base = []
                for i in range(eye_timelength):
                    timestamp = i * time_duration / eye_timelength
                    eye_timestamps_base.append(timestamp)

                pupil_l_base.append(pupil_l_base_seg)
                pupil_r_base.append(pupil_r_base_seg)
    new_participant.pupil_left_base, new_participant.pupil_right_base = pupil_l_base, pupil_r_base

    pupil_l_lightreflex = []
    pupil_r_lightreflex = []
    for i in range(0, len(pupil_l_base[0])):
        sum = 0
        for j in range(0, len(pupil_l_base)):
            sum += pupil_l_base[j][i]
        pupil_l_lightreflex.append(sum / len(pupil_l_base))

        sum = 0
        for j in range(0, len(pupil_r_base)):
            sum += pupil_r_base[j][i]
        pupil_r_lightreflex.append(sum / len(pupil_r_base))
    pupil_l_lightreflex = eye_data_filter(pupil_l_lightreflex, threshold = 4)
    pupil_r_lightreflex = eye_data_filter(pupil_r_lightreflex, threshold = 4)
    new_participant.pupil_left_lightreflex, new_participant.pupil_right_lightreflex = pupil_l_lightreflex, pupil_r_lightreflex



    # load eye data from VR set (pupil dilation & eye open degree)
    decouple_coefficient = 0.9
    time_duration = 5
    for i in range(1, 4):
        sub_file = participant_files_path + "\\" + str(i)

        # load pupil/eye total response in all tests, and decouple from light response
        for item in os.listdir(sub_file):
            if ((".csv" in item) & (item != "1.csv")):
                filename = sub_file + "\\" + item
                pupil_l_seg, pupil_r_seg, eye_l_seg, eye_r_seg = load_eye_data(filename, end_time=time_duration)

                # decouple from light reflex
                # print("seg len =",len(pupil_r_seg), ", base len =", len(pupil_r_base))
                pupil_l_seg = eye_data_filter(fix_blink(pupil_l_seg), threshold= 4)
                pupil_r_seg = eye_data_filter(fix_blink(pupil_r_seg), threshold= 4)
                pupil_r_seg_decouple = [pupil_r_seg[i] - decouple_coefficient * pupil_r_lightreflex[i] for i in range(0, len(pupil_r_lightreflex))]
                pupil_l_seg_decouple = [pupil_l_seg[i] - decouple_coefficient * pupil_l_lightreflex[i] for i in range(0, len(pupil_l_lightreflex))]

                # fix blink
                pupil_r_seg_final = eye_data_filter(pupil_r_seg_decouple)
                pupil_l_seg_final = eye_data_filter(pupil_l_seg_decouple)
                eye_r_seg_final = fix_blink(eye_r_seg)
                eye_l_seg_final = fix_blink(eye_l_seg)

                pupil_l.append(pupil_l_seg_final)
                pupil_r.append(pupil_r_seg_final)
                eye_l.append(eye_l_seg_final)
                eye_r.append(eye_r_seg_final)
                # print(new_participant.name, "pupil_r_seg:\n", pupil_r_seg)

                # load eye data timestamps
                eye_timelength = len(eye_l_seg)
                eye_timestamps_seg = []
                for i in range(eye_timelength):
                    timestamp = i * time_duration / eye_timelength
                    eye_timestamps_seg.append(timestamp)

                # print(new_participant.name, "eye_timestamps_seg length:\n", eye_timelength)
                eye_timestamps.append(eye_timestamps_seg)


    # attach data to a new_participant instance
    new_participant.timestamps, new_participant.raw_eda, new_participant.raw_ecg, new_participant.raw_ppg = \
        timestamps_, raw_eda_, raw_ecg_, raw_ppg_

    new_participant.pupil_left, new_participant.pupil_right = pupil_l, pupil_r
    new_participant.eye_left, new_participant.eye_right = eye_l, eye_r

    new_participant.eye_timestamps = eye_timestamps

    # maintain all participants(instances) in a participants list
    participants.append(new_participant)

print("Done loading!")
t2 = time.time()
print(f"{len(participants)} participants were loaded, costing {(t2 - t1)/len(participants)} s / person")

# display data
print("\nDisplaying Data ...")
time.sleep(0.5)

participant_num = len(participants)
participant_num_of_row = int(participant_num/4)+1

# plot eye data figures
plt.figure()
k2=0

for participant in participants:
    k2 += 1
    plt.subplot(participant_num_of_row, 4, k2)
    i = 0
    # print(participant.name, "eye records = ", len(participant.eye_left), "record len = ", len(participant.eye_left[i]))
    # print(participant.eye_timestamps[i])
    plt.plot(participant.eye_timestamps[i], participant.eye_left[i])
    plt.plot(participant.eye_timestamps[i], participant.eye_right[i])
    plt.plot(participant.eye_timestamps[i], participant.pupil_left[i])
    plt.plot(participant.eye_timestamps[i], participant.pupil_right[i])
    plt.xlabel("time")
    plt.ylabel("eye data")
    st1 = participant.name
    plt.title(st1)

plt.suptitle("EYE DATA")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.36, hspace=0.36)
plt.show()


# plot light reflex tests (3 segs)
plt.figure()
k3=0

for participant in participants:
    k3 += 1
    plt.subplot(participant_num_of_row, 4, k3)
    # print(participant.name, " base records = ", len(participant.pupil_left_base))
    for i in range(0,3):
        plt.plot(participant.eye_timestamps[i], participant.pupil_left_base[i])
        plt.plot(participant.eye_timestamps[i], participant.pupil_right_base[i])
    plt.xlabel("time")
    plt.ylabel("eye data")
    st1 = participant.name
    plt.title(st1)

plt.suptitle("LIGHT REFLEX DATA")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.36, hspace=0.36)
plt.show()


# plot average light reflex
plt.figure()
k4=0

for participant in participants:
    k4 += 1
    plt.subplot(participant_num_of_row, 4, k4)
    plt.plot(participant.eye_timestamps[i], participant.pupil_left_lightreflex)
    plt.plot(participant.eye_timestamps[i], participant.pupil_right_lightreflex)
    plt.xlabel("time")
    plt.ylabel("light reflex")
    st1 = participant.name
    plt.title(st1)

plt.suptitle("AVERAGE LIGHT REFLEX DATA")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.36, hspace=0.36)
plt.show()
print("Done displaying!")

print("\nProcessing data ...")
time.sleep(0.5)

print("Processing EDA data ...")
time.sleep(1)

# process EDA data and attach them to participants
count = 0
time_0 = time.time()
for participant in participants:
    participant.eda_processed = []
    participant.eda_phasic = []
    participant.eda_tonic = []
    participant.eda_normalized_phasic = []
    for i in range(len(participant.timestamps)):
        participant.eda_processed.append(eda_filter_normalize(participant.raw_eda[i], participant.timestamps[i]))
        time_total = participant.timestamps[i].values[-1]
        sample_delta = time_total / (len(participant.timestamps[i]) - 1)
        count += 1
        time_1 = time.time()
        print(
            f"Running cvxEDA ... {count}/{len(participants) * 30} , {(time_1 - time_0) * (len(participants) * 30 - count) / count} secs left\n")
        time.sleep(0.1)
        tonic, phasic = cvxEDA(participant.eda_processed[i], delta=sample_delta, tau0=2., tau1=0.7, delta_knot=10.,
                               alpha=8e-4, gamma=1e-2, solver=True, options={'reltol': 1e-9})
        participant.eda_phasic.append(phasic)
        participant.eda_tonic.append(tonic)
        #participant.eda_normalized_phasic.append((phasic-phasic.mean())/phasic.std())

print("Processing Eye data ...")
time.sleep(1)

# plot EDA figures

plt.figure()
k=0

for participant in participants:
    print("\n")
    print("Name: ",participant.name)
    k += 1
    for i in range(1):
        print("EDA phasic mean: ", participant.eda_phasic[i].mean())
        print("EDA phasic std:  ",participant.eda_phasic[i].std)
        print("EDA tonic mean:  ",participant.eda_tonic[i].mean)
        print("EDA tonic std:   ",participant.eda_tonic[i].std)
        plt.subplot(participant_num_of_row, 4, k)
        plt.plot(participant.timestamps[i],participant.eda_processed[i])
        plt.plot(participant.timestamps[i],participant.eda_phasic[i])
        plt.plot(participant.timestamps[i],participant.eda_tonic[i])
        plt.xlabel("time")
        plt.ylabel("eda data")
        plt.title(participant.name)

plt.suptitle("EDA DATA")
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.36, hspace=0.36)
plt.show()


print("Running feature extraction ...")

def feature_extraction_ecg(raw_ecg):
    features={}
    feature_list=[]
    ecg_clean=nk.ecg_process(raw_ecg, sampling_rate=64)[0]["ECG_Clean"]
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=64,show=False)
    rates=nk.ecg_rate(peaks, sampling_rate=64, desired_length=None, interpolation_method='monotone_cubic')
    features['heartrate_mean']=rates.mean()
    features['heartrate_std']=rates.std()
    hrv_features=nk.hrv_time(peaks, sampling_rate=64, show=False)
    features["hrv_MeanNN"]=hrv_features["HRV_MeanNN"][0]
    features["hrv_SDNN"]=hrv_features["HRV_SDNN"][0]
    features["hrv_RMSSD"]=hrv_features["HRV_RMSSD"][0]
    features["hrv_SDSD"]=hrv_features["HRV_SDSD"][0]
    features["hrv_MedianNN"]=hrv_features["HRV_MedianNN"][0]   
    features["hrv_Prc20NN"]=hrv_features["HRV_Prc20NN"][0]   
    features["hrv_Prc80NN"]=hrv_features["HRV_Prc80NN"][0]   
    features["hrv_pNN50"]=hrv_features["HRV_pNN50"][0]   
    features["hrv_pNN20"]=hrv_features["HRV_pNN20"][0]   
    features["hrv_MinNN"]=hrv_features["HRV_MinNN"][0]   
    features["hrv_MaxNN"]=hrv_features["HRV_MaxNN"][0]  
    feature_list=[value for value in features.values()]
    return(feature_list)

# extract ECG features
for participant in participants:
    ecg_features=[]
    for item in participant.raw_ecg:
        ecg_features.append(feature_extraction_ecg(item))
    print(f"{participant.name} ECG: {ecg_features}")
    ecg_features_df=pd.DataFrame(ecg_features)
    ecg_features_df.columns=['heartrate_mean','heartrate_std',"hrv_MeanNN","hrv_SDNN","hrv_RMSSD","hrv_SDSD","hrv_MedianNN","hrv_Prc20NN","hrv_Prc80NN","hrv_pNN50","hrv_pNN20","hrv_MinNN","hrv_MaxNN"]
    participant.features=ecg_features_df

# extract EDA features
for participant in participants:
    mean_phasic=[]
    mean_tonic=[]
    max_peaks=[]
    std_phasic=[]
    std_tonic=[]
    avr_amp=[]
    aucs=[]                             #area under curve
    aucs_raw=[]

    for i in range(len(participant.eda_phasic)):
        a,b=scipy.signal.find_peaks(participant.eda_phasic[i],height=0.1,distance=64)
        if (len(b["peak_heights"])>0):
            max_peaks.append(b["peak_heights"].max())
            avr_amp.append(b["peak_heights"].mean())
        else:
            max_peaks.append(0)
            avr_amp.append(0)
        mean_phasic.append(participant.eda_phasic[i].mean())
        std_phasic.append(participant.eda_phasic[i].std())
        mean_tonic.append(participant.eda_tonic[i].mean())
        std_tonic.append(participant.eda_tonic[i].std())
        aucs.append(auc(participant.timestamps[i],participant.eda_phasic[i]))
        aucs_raw.append(auc(participant.timestamps[i],participant.raw_eda[i]))

    df = pd.DataFrame(
        {"max_peak_height": max_peaks, "avr_peak_amp": avr_amp, "phasic_mean": mean_phasic, "phasic_std": std_phasic,
         "tonic_mean": mean_tonic, "tonic_std": std_tonic, "phasic_auc": aucs, "raw_auc": aucs_raw})
    participant.features = pd.concat([participant.features, df], axis=1)


# extract eye features
"""
    Extracted eye features:
    eye feature 1-5 : pupil dilation (light reflex decoupled) at 0-0.2s, 0.2-0.4s, 0.4-0.6s, 0.6-0.8s, 0.8-1.0s
    eye feature 6: pupil dilation change : 0-0.4s(avg) - 0.6-1.0(avg)
    eye feature 7: pupil dilation std
    eye feature 8: eye openess mean
    eye feature 9: eye openess std
    eye feature 10: blinking frequency (not added yet)
"""
for participant in participants:
    # initialize all features
    pupil_0, pupil_1, pupil_2, pupil_3, pupil_4 = [], [], [], [], []
    pupil_change = []
    pupil_std = []
    eye_mean = []
    eye_std = []

    for i in range(0, len(participant.pupil_left)):
        # extract pupil features
        pupil = []
        for j in range(0, len(participant.pupil_left[i])):
            pupil.append(0.5 * (participant.pupil_left[i][j] + participant.pupil_right[i][j]))
        len_seg = 90
        pupil_0.append(np.mean(pupil[int(0*len_seg) : int(0.2*len_seg)]))
        pupil_1.append(np.mean(pupil[int(0.2 * len_seg): int(0.4 * len_seg)]))
        pupil_2.append(np.mean(pupil[int(0.4 * len_seg): int(0.6 * len_seg)]))
        pupil_3.append(np.mean(pupil[int(0.6 * len_seg): int(0.8 * len_seg)]))
        pupil_4.append(np.mean(pupil[int(0.8 * len_seg): int(1 * len_seg)]))
        pupil_change.append(np.mean(pupil[0 : int(0.4*len_seg)]) - np.mean(pupil[int(0.6 * len_seg) : int(1*len_seg)]))
        pupil_std.append(np.std(pupil))

        # extract eye features
        eyes = []
        for j in range(0, len(participant.eye_left[i])):
            eyes.append(0.5 * (participant.eye_left[i][j] + participant.eye_right[i][j]))
        eye_mean.append(np.mean(eyes))
        eye_std.append(np.std(eyes))

    df_eye = pd.DataFrame(
        {"pupil 0-0.2": pupil_0, "pupil 0.2-0.4": pupil_1, "pupil 0.4-0.6": pupil_2, "pupil 0.6-0.8": pupil_3, "pupil 0.8-1": pupil_4,
         "pupil_change 0-1" : pupil_change, "pupil_std" : pupil_std, "eye_mean" : eye_mean, "eye_std" : eye_std})
    participant.features = pd.concat([participant.features, df_eye], axis=1)


    # print(ecg_features)
    ecg_features_df=pd.DataFrame(ecg_features)
    ecg_features_df.columns=['heartrate_mean','heartrate_std',"hrv_MeanNN","hrv_SDNN","hrv_RMSSD","hrv_SDSD","hrv_MedianNN","hrv_Prc20NN","hrv_Prc80NN","hrv_pNN50","hrv_pNN20","hrv_MinNN","hrv_MaxNN"]
    df_eye = pd.DataFrame(
        {"max_peak_height": max_peaks, "avr_peak_amp": avr_amp, "phasic_mean": mean_phasic, "phasic_std": std_phasic,
         "tonic_mean": mean_tonic, "tonic_std": std_tonic, "phasic_auc": aucs, "raw_auc": aucs_raw})
    participant.features=pd.concat([participant.features, df_eye], axis=1)


# Displaying 1 participants features
# print("Feature sampling: \n", participants[3].features)

features_total=participants[0].features
labels_total=participants[0].labels
for i in range(1,len(participants)):
    a=participants[i].features.copy()
    features_total=features_total.append(a)
    labels_total=labels_total.append(participants[i].labels.copy()) 
features_total=features_total.reset_index(drop=True)
labels_total=labels_total.reset_index(drop=True)
features_ecg=features_total.iloc[:,:-7]            
features_eda=features_total.iloc[:,-7:]                                  
features_csv=pd.concat([features_total,labels_total],axis=1).to_csv(r"C:\VR_DATA\features.csv")


# Training model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

# In[ ]:

print("\nRunning RandomForestRegressor (indiv. seperately) ...")
mses=[]
for participant in participants:   
    x_train,x_test,y_train,y_test=train_test_split(participant.features,participant.labels,train_size=24)
    reg=RandomForestRegressor()
    reg.fit(x_train,y_train)
    predicts=reg.predict(x_test)
   # print(pd.DataFrame([y_test.to_list(),predicts]))
    mses.append(mean_squared_error(predicts,y_test))
#pd.DataFrame(y_test.tolist(),predicts).to_csv(r"C:/Autumn2022/initial_result.csv")
mses=np.array(mses)
print(f"\nRMSES (default, indiv. sep.: \n{mses**0.5}")
print(f"RMSE mean (default, indiv. sep.): {(mses**0.5).mean()}")

# In[ ]:

# RandomForestRegressor
print("\nRunning RandomForestRegressor ...")
acc = []
rmse = []
rmse_all = []
for i in range(0,50):
    x_train, x_test, y_train, y_test = train_test_split(features_total, labels_total, train_size=200)
    reg2 = RandomForestRegressor(max_depth=50,
                                 max_features=1,
                                 min_impurity_decrease=0,
                                 min_samples_leaf=1)
    reg2.fit(x_train, y_train)
    predicts = reg2.predict(x_test)
    # print(pd.DataFrame([y_test.to_list(),predicts]))
    # print(mean_squared_error(predicts,y_test)**0.5)
    mean_predictor = np.full(len(predicts), 4)
    labels_total.median()
    fear_labels = np.array(y_test > 4)
    fear_predictions = np.array(predicts > 4)
    accuracy = (fear_labels == fear_predictions).mean()
    acc.append(accuracy)
    rmse_scores = cross_val_score(
        reg2, features_total, labels_total, cv=5, scoring='neg_root_mean_squared_error')
    rmse_mean = rmse_scores.mean()
    rmse.append(rmse_mean)
    rmse_all.append(rmse_scores)
print(f"Random Forest binary classification accuracy: {statistics.mean(acc)}")
print("RandomForestRegressor2 5-fold cross validation:")
print(f"RMSE mean: {statistics.mean(rmse)}")
print(f"RMSE: {rmse_all}")


# LinearRegression
print("\nRunning LinearRegression...")
acc2 = []
rmse2 = []
rmse_all2 = []
for i in range(0,50):
    x_train, x_test, y_train, y_test = train_test_split(features_total, labels_total, train_size=200)
    reg2 = linear_model.LinearRegression()
    reg2.fit(x_train, y_train)
    predicts = reg2.predict(x_test)
    # print(pd.DataFrame([y_test.to_list(),predicts]))
    # print(f"RMSE_{i}: {mean_squared_error(predicts, y_test) ** 0.5}")
    mean_predictor = np.full(len(predicts), 4)
    labels_total.median()
    fear_labels = np.array(y_test > 4)
    fear_predictions = np.array(predicts > 4)
    accuracy2 = (fear_labels == fear_predictions).mean()
    acc2.append(accuracy2)
    rmse_scores = cross_val_score(
        reg2, features_total, labels_total, cv=5, scoring='neg_root_mean_squared_error')
    rmse_mean2 = rmse_scores.mean()
    rmse2.append(rmse_mean2)
    rmse_all2.append(rmse_scores)
print(f"Linear Regression binary classification accuracy: {statistics.mean(acc2)}")
print("5-fold cross validation:")
print(f"RMSE mean: {statistics.mean(rmse2)}")
print(f"RMSE: {rmse_all2}")






