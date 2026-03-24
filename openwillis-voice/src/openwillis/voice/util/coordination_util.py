"""Functions below are taken from SSL-SI-tool repository:
https://github.com/Yashish92/SSL-SI-tool/tree/master
and are modified for efficiency and speed
"""
# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

import os

import pandas as pd
import numpy as np
import librosa
import torch

from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from tensorflow.keras.models import load_model


FS = 16000

def KalmanRTSSmoother(MMf, MMp, PPf, PPp, A):
    """
    ------------------------------------------------------------------------------------------------------
    Perform Kalman Filter prediction step. The model is:

        x[k] = A*x[k-1] + Bw[k-1],  w ~ N(0,Q).
        y[k] = C*x[k]   + v[k]      v ~ N(0,R)

    Example:
        Y is measurement
        x = x0
        P = P0
        [MMf, MMp, PPf, PPp] = KalmanLoop(x, P, A, B, Y, Q, R, D, u)
        [MMs, PPs] = KalmanRTSSmoother(MMf, MMp, PPf, PPp, A)

    Parameters:
    ...........
    MMf: numpy.ndarray
        Forward filtered state estimates.
    MMp: numpy.ndarray
        Forward predicted state estimates.
    PPf: list
        Forward filtered state covariance matrices.
    PPp: list
        Forward predicted state covariance matrices.
    A: numpy.ndarray
        State transition matrix.

    Returns:
    ...........
    MMs: numpy.ndarray
        Smoothed state estimates.
    PPs: list
        Smoothed state covariance matrices.
    ------------------------------------------------------------------------------------------------------
    """
    N = MMf.shape[1]
    N1 = MMf.shape[0]

    MMs = np.zeros((N1,N), dtype=float)
    MMs[:,-1] = MMf[:,-1]
    PPs = [None]*N
    PPs[N-1] = PPf[-1]
    Xs = MMf[:,-1]
    Ps = PPf[-1]

    for i in range(N-2, 0, -1):
        Xf = MMf[:,i]
        Pf = PPf[i]
        
        Xp = MMp[:,i+1]
        Pp = PPp[i+1]

        J = np.dot(np.dot(Pf, A.T), np.linalg.inv(Pp))
        Xs = Xf + np.dot(J, (Xs-Xp))
        Ps = Pf + np.dot(np.dot(J, (Ps-Pp)), J.T)
        MMs[:,i] = Xs
        PPs[i] = Ps

    return MMs, PPs
    
def KalmanUpdate(X, P, m, C, R):
    """
    ------------------------------------------------------------------------------------------------------
    Kalman Filter update step.

    Perform Kalman Filter prediction step. The model is:

        x[k] = A*x[k-1] + Bw[k-1],  w ~ N(0,Q).
        y[k] = C*x[k]   + v[k],     v ~ N(0,R)

    Parameters:
    ...........
    X: numpy.ndarray
        State estimate.
    P: numpy.ndarray
        State covariance matrix.
    m: numpy.ndarray
        Measurement.
    C: numpy.ndarray
        Measurement matrix.
    R: numpy.ndarray
        Measurement noise covariance matrix.

    Returns:
    ...........
    X: numpy.ndarray
        Updated state estimate.
    P: numpy.ndarray
        Updated state covariance matrix.
    ------------------------------------------------------------------------------------------------------
    """
# update step
    z = np.dot(C, X)
    v = m - z

    S = np.dot(np.dot(C, P), C.T) + R # C * P * C' + R
    K = np.dot(np.dot(P, C.T), np.linalg.inv(S)) # P * C' * inv(S)
    X += + np.dot(K, v)
    P -= np.dot(np.dot(K, S), K.T)

    return X,P

def KalmanPredict(x, P, A, B, Q):
    """
    ------------------------------------------------------------------------------------------------------
    Kalman Filter prediction step.

    Perform Kalman Filter prediction step. The model is:

        x[k] = A*x[k-1] + Bw[k-1],  w ~ N(0,Q).
        y[k] = C*x[k]   + v[k],     v ~ N(0,R)

    Parameters:
    ...........
    x: numpy.ndarray
        State estimate.
    P: numpy.ndarray
        State covariance matrix.
    A: numpy.ndarray
        State transition matrix.
    B: numpy.ndarray
        Control input matrix.
    Q: numpy.ndarray
        Process noise covariance matrix.

    Returns:
    ...........
    x: numpy.ndarray
        Predicted state estimate.
    P: numpy.ndarray
        Predicted state covariance matrix.
    ------------------------------------------------------------------------------------------------------
    """
    x = np.dot(A,x)
    P = np.dot(np.dot(A, P), A.T) + np.dot(np.dot(B, Q), B.T) # A * P * A' + B * Q * B'
    return x,P 
    
def KalmanLoop(x, P, A, B, C, Y, Q, R):
    """
    ------------------------------------------------------------------------------------------------------
    Perform Kalman Filter prediction and update steps in a loop.

    The model is:

        x[k] = A*x[k-1] + Bw[k-1],  w ~ N(0,Q).
        y[k] = C*x[k]   + v[k],     v ~ N(0,R)

    Parameters:
    ...........
    x: numpy.ndarray
        Initial state estimate.
    P: numpy.ndarray
        Initial state covariance matrix.
    A: numpy.ndarray
        State transition matrix.
    B: numpy.ndarray
        Control input matrix.
    C: numpy.ndarray
        Measurement matrix.
    Y: numpy.ndarray
        Measurements.
    Q: numpy.ndarray
        Process noise covariance matrix.
    R: numpy.ndarray
        Measurement noise covariance matrix.

    Returns:
    ...........
    MMf: numpy.ndarray
        Forward filtered state estimates.
    MMp: numpy.ndarray
        Forward predicted state estimates.
    PPf: list
        Forward filtered state covariance matrices.
    PPp: list
        Forward predicted state covariance matrices.
    ------------------------------------------------------------------------------------------------------
    """
    N = Y.shape[1] #Number of Measurements

    MMf = np.zeros((2, N))
    PPf = [] #cell(1,N);
    MMp = np.zeros((2, N))
    PPp = [] #cell(1,N)

    # Perform prediction
    for i in range(N):
        (x,P) = KalmanPredict(x, P, A, B, Q)
        MMp[:, i, None] = x
        PPp.append(P)
        (x,P) = KalmanUpdate(x, P, Y[0, i], C, R)
        MMf[:, i, None] = x
        PPf.append(P)

    return MMf, MMp, PPf, PPp

def kalmansmooth(tst_trg, R=0.25):
    """
    ------------------------------------------------------------------------------------------------------

    Perform Kalman smoothing on the input data.

    Parameters:
    ...........
    tst_trg: numpy.ndarray
        The input data.
    R: float
        Smoothness factor.

    Returns:
    ...........
    tst_trg_sm: numpy.ndarray
        The smoothed data.

    ------------------------------------------------------------------------------------------------------
    """
    data_sm = []

    for iter1 in range(tst_trg.shape[0]):
        EstdData = tst_trg[None, iter1, :]

        Q = 300
        T = 0.1
        A = np.asarray([[1, T], [0, 1]])
        B = np.asarray([[(T**2)/2], [T]])
        C = np.asarray([[1, 0]])
        P0= np.eye(2)
        x0= np.asarray([[EstdData[0, 0]],[0]])
        x = x0
        P = P0

        (MMf, MMp, PPf, PPp) = KalmanLoop(x, P, A, B, C, EstdData, Q, R)
        MMs, _ = KalmanRTSSmoother(MMf, MMp, PPf, PPp, A)

        data_sm.append(MMs[0, :, None])

    tst_trg_sm = np.concatenate(data_sm, axis=1)
    return tst_trg_sm.T

def feature_extract(wav_file):
    """
    ------------------------------------------------------------------------------------------------------

    Extracts the speech representations from the input audio file using the HuBERT model.

    Parameters:
    ...........
    wav_file: str
        The path to the audio file.

    Returns:
    ...........
    spk_wav_ssl_npy: numpy.ndarray
        The extracted speech representations.
    no_segs: int
        The number of segments the audio file was divided into.
    file_len: int
        The length of the audio file.

    ------------------------------------------------------------------------------------------------------
    """
    maxlen = 2  # seconds
    model_extractor = HuBERT("facebook/hubert-large-ll60k", save_path='')

    source_ar, sr = librosa.load(wav_file, sr=FS)

    audio_len = int(maxlen * sr)
    file_len = len(source_ar)

    first_test = True
    # If the audio file is shorter than the maximum length - 2 seconds
    if len(source_ar) <= audio_len:
        no_segs = 1
        pad_amt = audio_len - len(source_ar)
        spk_wav = np.concatenate([source_ar, np.zeros(pad_amt, dtype=np.float32)])
        spk_wav_tensor = torch.from_numpy(spk_wav)
        spk_wav_tensor_un = spk_wav_tensor.unsqueeze(0)
        spk_wav_ssl = model_extractor(spk_wav_tensor_un)
        spk_wav_ssl = spk_wav_ssl.squeeze(0)

        spk_wav_ssl_npy = spk_wav_ssl.detach().numpy()
        spk_wav_ssl_npy = np.pad(spk_wav_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
        spk_wav_ssl_npy = np.expand_dims(spk_wav_ssl_npy, axis=0)

    # If the audio file is longer than the maximum length and needs to be divided into segments
    elif len(source_ar) > audio_len:

        no_segs = len(source_ar) // audio_len + 1
        for i in range(0, no_segs):
            if first_test:
                spk_data_seg = source_ar[:audio_len]
                spk_seg_tensor = torch.from_numpy(spk_data_seg)
                spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                spk_seg_ssl = model_extractor(spk_seg_tensor_un)

                spk_seg_ssl = spk_seg_ssl.squeeze(0)
                spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                spk_wav_ssl_npy = spk_seg_ssl_npy
                first_test = False
            elif i < no_segs - 1:
                spk_data_seg = source_ar[audio_len * i:audio_len * (i + 1)]
                spk_seg_tensor = torch.from_numpy(spk_data_seg)
                spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                spk_seg_ssl = model_extractor(spk_seg_tensor_un)

                spk_seg_ssl = spk_seg_ssl.squeeze(0)
                spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                spk_wav_ssl_npy = np.vstack((spk_wav_ssl_npy, spk_seg_ssl_npy))
            elif i == no_segs - 1:
                pad_amt = (audio_len * no_segs) - file_len
                spk_data_seg = np.concatenate([source_ar[(audio_len * i):], np.zeros(pad_amt, dtype=np.float32)])
                spk_seg_tensor = torch.from_numpy(spk_data_seg)
                spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                spk_seg_ssl = model_extractor(spk_seg_tensor_un)

                spk_seg_ssl = spk_seg_ssl.squeeze(0)
                spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                spk_wav_ssl_npy = np.vstack((spk_wav_ssl_npy, spk_seg_ssl_npy))

    return spk_wav_ssl_npy, no_segs, file_len

def create_time_delay_matrix(tv_matrix, delay, num_delays):
    """
    ------------------------------------------------------------------------------------------------------

    Constructs a time-delay embedded matrix where each TV channel is expanded
    with its delayed versions.

    Parameters:
    ...........
    tv_matrix: numpy.ndarray
        The input matrix.
    delay: int
        The delay value.
    num_delays: int
        The number of delays.

    Returns:
    ...........
    time_delay_matrix: numpy.ndarray
        The time-delay embedded matrix.

    ------------------------------------------------------------------------------------------------------
    """
    num_tvs, num_timepoints = tv_matrix.shape
    time_delay_matrix = np.zeros((num_tvs * num_delays, num_timepoints - delay * (num_delays - 1)))

    for i in range(num_delays):
        time_delay_matrix[i * num_tvs:(i + 1) * num_tvs, :] = tv_matrix[:, i * delay:i * delay + time_delay_matrix.shape[1]]

    return time_delay_matrix


def calculate_articulation_coordination(audio_path, measures):
    """
    ------------------------------------------------------------------------------------------------------

    Calculates the articulation coordination features of the input audio file.

    Parameters:
    ...........
    audio_path: str
        The path to the audio file.
    measures: dict
        The dictionary containing the feature names.

    Returns:
    ...........
    df_articulation: pandas.DataFrame
        The articulation coordination features.

    ------------------------------------------------------------------------------------------------------
    """
    feature_data, _, audio_len = feature_extract(audio_path)

    loaded_model = load_model(os.path.join(os.path.dirname(__file__), 'models/BLSTM_model.h5'))
    y_predict = loaded_model.predict(feature_data, verbose=0)

    for i in range(0, y_predict.shape[0]):
        seg_TVs = y_predict[i]
        if i == 0:
            final_TVs = seg_TVs
        else:
            final_TVs = np.concatenate((final_TVs, seg_TVs), axis=0)

    audio_time = audio_len / FS
    TV_len = int(audio_time * 100)  # TV sampling rate is 100Hz

    final_TVs = np.transpose(final_TVs)
    final_TVs = final_TVs[:, 0:TV_len]

    tv_smth = kalmansmooth(final_TVs)
    time_delay_embedded = create_time_delay_matrix(tv_smth, 7, 15)
    correlation_matrix = np.corrcoef(time_delay_embedded)
    eigenvalues, _ = np.linalg.eigh(correlation_matrix)
    eigenspectrum = np.sort(eigenvalues)[::-1]

    acf1 = np.mean(eigenspectrum[:5])
    acf2 = np.mean(eigenspectrum[42:47])
    acf3 = np.mean(eigenspectrum[-5:])

    df_articulation =  pd.DataFrame({measures['ACF1']: [acf1], measures['ACF2']: [acf2], measures['ACF3']: [acf3]})
    return df_articulation
