import os
import numpy as np

from tqdm import *
from numpy.linalg import norm
from neurokit import (ecg_preprocess, ecg_hrv,
                      ecg_signal_quality)


def PQRST(ecg_prep):
    ecg_feats = ecg_prep['ECG']
    R_Peaks = ecg_feats['R_Peaks']
    T_Waves = ecg_feats['T_Waves']
    Q_Waves = ecg_feats['Q_Waves']
    P_Waves = ecg_feats['P_Waves']
    S_Waves = ecg_feats['S_Waves']

    pqrst = []
    for r in R_Peaks:
        tmp = [-1] * 5
        tmp[2] = r
        r_left = r - 200
        r_right = r + 200

        for p in P_Waves:
            if r_left <= p < r:
                tmp[0] = p
                # break

        for q in Q_Waves:
            if r_left < q < r:
                tmp[1] = q
                # break

        for s in S_Waves:
            if r < s < r_right:
                tmp[3] = s
                break

        for t in T_Waves:
            if r < t <= r_right:
                tmp[4] = t
                break

        pqrst.append(tmp)
    return pqrst


def time_feats(pqrst):

    def mean_time(time):
        mt = np.nan
        if len(time) > 0:
            mt = np.mean(time) / 500
        return mt

    PQs, PRs, PSs, PTs = [], [], [], []
    QRs, QSs, QTs = [], [], []
    RSs, RTs = [], []
    STs = []

    for p, q, r, s, t in pqrst:
        if p != -1:
            if q != -1:
                PQs.append(q - p)
            if r != -1:
                PRs.append(r - p)
            if s != -1:
                PSs.append(s - p)
            if t != -1:
                PTs.append(t - p)
        if q != -1:
            if r != -1:
                QRs.append(r - q)
            if s != -1:
                QSs.append(s - q)
            if t != -1:
                QTs.append(t - q)
        if r != -1:
            if s != -1:
                RSs.append(s - r)
            if t != -1:
                RTs.append(t - r)
        if s != -1:
            if t != -1:
                STs.append(t - s)

    PQ = mean_time(PQs)
    PR = mean_time(PRs)
    PS = mean_time(PSs)
    PT = mean_time(PTs)
    QR = mean_time(QRs)
    QS = mean_time(QSs)
    QT = mean_time(QTs)
    RS = mean_time(RSs)
    RT = mean_time(RTs)
    ST = mean_time(STs)

    PT_QS, QT_QS = np.nan, np.nan
    if QS is not np.nan and QS > 0:
        if PT is not np.nan and PT > 0:
            PT_QS = PT / QS
        if QT is not np.nan and QT > 0:
            QT_QS = QT / QS

    time = [
        PQ, PR, PS, PT, QR, QS, QT,
        RS, RT, ST, PT_QS, QT_QS
    ]
    return time


def amplitude_feats(ecg, pqrst):

    def mean_amplitude(amplitude):
        ma = np.nan
        if len(amplitude) > 0:
            ma = np.mean(amplitude)
        return ma

    Pys, Qys, Rys, Sys, Tys = [], [], [], [], []
    PQs, PRs, PSs, PTs = [], [], [], []
    QRs, QSs, QTs = [], [], []
    RSs, RTs = [], []
    STs = []

    for p, q, r, s, t in pqrst:
        if p != -1:
            Pys.append(ecg[p])
            if q != -1:
                PQs.append(ecg[q] - ecg[p])
            if r != -1:
                PRs.append(ecg[r] - ecg[p])
            if s != -1:
                PSs.append(ecg[s] - ecg[p])
            if t != -1:
                PTs.append(ecg[t] - ecg[p])
        if q != -1:
            Qys.append(ecg[q])
            if r != -1:
                QRs.append(ecg[r] - ecg[q])
            if s != -1:
                QSs.append(ecg[s] - ecg[q])
            if t != -1:
                QTs.append(ecg[t] - ecg[q])
        if r != -1:
            Rys.append(ecg[r])
            if s != -1:
                RSs.append(ecg[s] - ecg[r])
            if t != -1:
                RTs.append(ecg[t] - ecg[r])
        if s != -1:
            Sys.append(ecg[s])
            if t != -1:
                STs.append(ecg[t] - ecg[s])
        if t != -1:
            Tys.append(ecg[t])

    Py = mean_amplitude(Pys)
    Qy = mean_amplitude(Qys)
    Ry = mean_amplitude(Rys)
    Sy = mean_amplitude(Sys)
    Ty = mean_amplitude(Tys)
    PQ = mean_amplitude(PQs)
    PR = mean_amplitude(PRs)
    PS = mean_amplitude(PSs)
    PT = mean_amplitude(PTs)
    QR = mean_amplitude(QRs)
    QS = mean_amplitude(QSs)
    QT = mean_amplitude(QTs)
    RS = mean_amplitude(RSs)
    RT = mean_amplitude(RTs)
    ST = mean_amplitude(STs)

    PQ_QR, PQ_QS, PQ_QT, PQ_PS, PQ_RS = \
        np.nan, np.nan, np.nan, np.nan, np.nan
    if PQ is not np.nan:
        if QR is not np.nan and QR != 0:
            PQ_QR = PQ / QR
        if QS is not np.nan and QS != 0:
            PQ_QS = PQ / QS
        if QT is not np.nan and QT != 0:
            PQ_QT = PQ / QT
        if PS is not np.nan and PS != 0:
            PQ_PS = PQ / PS
        if RS is not np.nan and RS != 0:
            PQ_RS = PQ / RS

    RS_QR, RS_QS, RS_QT = np.nan, np.nan, np.nan
    if RS is not np.nan:
        if QR is not np.nan and QR != 0:
            RS_QR = RS / QR
        if QS is not np.nan and QS != 0:
            RS_QS = RS / QS
        if QT is not np.nan and QT != 0:
            RS_QT = RS / QT

    ST_QS, ST_PQ, ST_QT = np.nan, np.nan, np.nan
    if ST is not np.nan:
        if QS is not np.nan and QS != 0:
            ST_QS = ST / QS
        if PQ is not np.nan and PQ != 0:
            ST_PQ = ST / PQ
        if QT is not np.nan and QT != 0:
            ST_QT = ST / QT

    amplitude = [
        Py, Qy, Ry, Sy, Ty,
        PQ, PR, PS, PT, QR,
        QS, QT, RS, RT, ST,
        PQ_QR, PQ_QS, PQ_QT,
        PQ_PS, PQ_RS, RS_QR,
        RS_QS, RS_QT, ST_QS,
        ST_PQ, ST_QT
    ]
    return amplitude


def distance_feats(ecg, pqrst):

    def mean_distance(distance):
        md = np.nan
        if len(distance) > 0:
            md = np.mean(distance)
        return md

    PQs, PRs, PSs, PTs = [], [], [], []
    QRs, QSs, QTs = [], [], []
    RSs, RTs = [], []
    STs = []

    for p, q, r, s, t in pqrst:
        if p != -1:
            pp = np.array([p / 500, ecg[p]])
            if q != -1:
                qp = np.array([q / 500, ecg[q]])
                PQs.append(norm(qp - pp))
            if r != -1:
                rp = np.array([r / 500, ecg[r]])
                PRs.append(norm(rp - pp))
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                PSs.append(norm(sp - pp))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                PTs.append(norm(tp - pp))

        if q != -1:
            qp = np.array([q / 500, ecg[q]])
            if r != -1:
                rp = np.array([r / 500, ecg[r]])
                QRs.append(norm(rp - qp))
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                QSs.append(norm(sp - qp))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                QTs.append(norm(tp - qp))

        if r != -1:
            rp = np.array([r / 500, ecg[r]])
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                RSs.append(norm(sp - rp))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                RTs.append(norm(tp - rp))

        if s != -1:
            sp = np.array([s / 500, ecg[s]])
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                STs.append(norm(tp - sp))

    PQ = mean_distance(PQs)
    PR = mean_distance(PRs)
    PS = mean_distance(PSs)
    PT = mean_distance(PTs)
    QR = mean_distance(QRs)
    QS = mean_distance(QSs)
    QT = mean_distance(QTs)
    RS = mean_distance(RSs)
    RT = mean_distance(RTs)
    ST = mean_distance(STs)

    ST_QS, RS_QR = np.nan, np.nan
    if QS is not np.nan and QS > 0:
        if ST is not np.nan:
            ST_QS = ST / QS
    if QR is not np.nan and QR > 0:
        if RS is not np.nan:
            RS_QR = RS / QR

    distance = [
        PQ, PR, PS, PT, QR, QS, QT,
        RS, RT, ST, ST_QS, RS_QR
    ]
    return distance


def slope_feats(ecg, pqrst):

    def mean_slope(slope):
        ms = np.nan
        if len(slope) > 0:
            ms = np.mean(slope)
        return ms

    PQs, PRs, PSs, PTs = [], [], [], []
    QRs, QSs, QTs = [], [], []
    RSs, RTs = [], []
    STs = []

    for p, q, r, s, t in pqrst:
        if p != -1:
            pp = np.array([p / 500, ecg[p]])
            if q != -1:
                qp = np.array([q / 500, ecg[q]])
                PQs.append((pp[1] - qp[1]) / (pp[0] - qp[0]))
            if r != -1:
                rp = np.array([r / 500, ecg[r]])
                PRs.append((pp[1] - rp[1]) / (pp[0] - rp[0]))
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                PSs.append((pp[1] - sp[1]) / (pp[0] - sp[0]))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                PTs.append((pp[1] - tp[1]) / (pp[0] - tp[0]))

        if q != -1:
            qp = np.array([q / 500, ecg[q]])
            if r != -1:
                rp = np.array([r / 500, ecg[r]])
                QRs.append((qp[1] - rp[1]) / (qp[0] - rp[0]))
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                QSs.append((qp[1] - sp[1]) / (qp[0] - sp[0]))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                QTs.append((qp[1] - tp[1]) / (qp[0] - tp[0]))

        if r != -1:
            rp = np.array([r / 500, ecg[r]])
            if s != -1:
                sp = np.array([s / 500, ecg[s]])
                RSs.append((rp[1] - sp[1]) / (rp[0] - sp[0]))
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                RTs.append((rp[1] - tp[1]) / (rp[0] - tp[0]))

        if s != -1:
            sp = np.array([s / 500, ecg[s]])
            if t != -1:
                tp = np.array([t / 500, ecg[t]])
                STs.append((sp[1] - tp[1]) / (sp[0] - tp[0]))

    PQ = mean_slope(PQs)
    PR = mean_slope(PRs)
    PS = mean_slope(PSs)
    PT = mean_slope(PTs)
    QR = mean_slope(QRs)
    QS = mean_slope(QSs)
    QT = mean_slope(QTs)
    RS = mean_slope(RSs)
    RT = mean_slope(RTs)
    ST = mean_slope(STs)

    slope = [PQ, PR, PS, PT, QR,
             QS, QT, RS, RT, ST]
    return slope


def angle_feats(ecg, pqrst):

    def compute_angle(p1, p2, p3):
        p1 = np.array(p1) if not isinstance(p1, np.ndarray) else p1
        p2 = np.array(p2) if not isinstance(p2, np.ndarray) else p2
        p3 = np.array(p3) if not isinstance(p3, np.ndarray) else p3

        p12 = p1 - p2
        p32 = p3 - p2

        cos_angle = np.dot(p12, p32) / (norm(p12) * norm(p32))
        angle = np.arccos(cos_angle)
        return angle

    def mean_angle(angle):
        ma = np.nan
        if len(angle) > 0:
            ma = np.mean(angle)
        return ma

    QRSs, RQSs, RSQs, RSTs, RTSs, TRSs, PQRs, PRQs, RPQs = \
        [], [], [], [], [], [], [], [], []

    for p, q, r, s, t in pqrst:
        if q != -1 and r != -1 and s != -1:
            Q = [q / 500, ecg[q]]
            R = [r / 500, ecg[r]]
            S = [s / 500, ecg[s]]
            QRSs.append(compute_angle(Q, R, S))
            RQSs.append(compute_angle(R, Q, S))
            RSQs.append(compute_angle(R, S, Q))

        if r != -1 and s != -1 and t != -1:
            R = [r / 500, ecg[r]]
            S = [s / 500, ecg[s]]
            T = [t / 500, ecg[t]]
            RSTs.append(compute_angle(R, S, T))
            RTSs.append(compute_angle(R, T, S))
            TRSs.append(compute_angle(T, R, S))

        if r != -1 and q != -1 and p != -1:
            R = [r / 500, ecg[r]]
            Q = [q / 500, ecg[q]]
            P = [p / 500, ecg[p]]
            PQRs.append(compute_angle(P, Q, R))
            PRQs.append(compute_angle(P, R, Q))
            RPQs.append(compute_angle(R, P, Q))

    QRS = mean_angle(QRSs)
    RQS = mean_angle(RQSs)
    RSQ = mean_angle(RSQs)
    RST = mean_angle(RSTs)
    RTS = mean_angle(RTSs)
    TRS = mean_angle(TRSs)
    PQR = mean_angle(PQRs)
    PRQ = mean_angle(PRQs)
    RPQ = mean_angle(RPQs)

    R_S, R_Q, R_T, R_P = \
        np.nan, np.nan, np.nan, np.nan
    Q_P, Q_T, S_P, S_T = \
        np.nan, np.nan, np.nan, np.nan

    if QRS is not np.nan:
        if RST is not np.nan and RST > 0:
            R_S = QRS / RST
        if PQR is not np.nan and PQR > 0:
            R_Q = QRS / PQR
        if RTS is not np.nan and RTS > 0:
            R_T = QRS / RTS
        if RPQ is not np.nan and RPQ > 0:
            R_P = QRS / RPQ

    if PQR is not np.nan:
        if RPQ is not np.nan and RPQ > 0:
            Q_P = PQR / RPQ
        if RTS is not np.nan and RTS > 0:
            Q_T = PQR / RTS

    if RST is not np.nan:
        if RPQ is not np.nan and RPQ > 0:
            S_P = RST / RPQ
        if RTS is not np.nan and RTS > 0:
            S_T = RST / RTS

    angle = [
        QRS, RQS, RSQ,
        RST, RTS, TRS,
        PQR, PRQ, RPQ,
        R_S, R_Q, R_T, R_P,
        Q_P, Q_T, S_P, S_T
    ]
    return angle


# def miscellaneous_feats(ecg, pqrst, time, amplitude,
#                         distance, slope, angle):

#     # Area
#     QRS_areas, PRQ_areas, SPT_areas = [], [], []
#     # Perimeter
#     QRS_pmts, PRQ_pmts, SPT_pmts = [], [], []
#     # Others

#     return


def quality_feats(quality_dict):
    cycles_quality = quality_dict['Cardiac_Cycles_Signal_Quality']
    ecg_quality = quality_dict['ECG_Signal_Quality']

    cycles_quality_stats = [
        np.mean(cycles_quality),
        np.std(cycles_quality),
        np.max(cycles_quality),
        np.min(cycles_quality),
        np.median(cycles_quality)
    ]

    ecg_quality_stats = [
        np.mean(ecg_quality),
        np.std(ecg_quality),
        np.max(ecg_quality),
        np.min(ecg_quality),
        np.median(ecg_quality)
    ]
    return cycles_quality_stats + ecg_quality_stats


def hrv_feats(hrv_dict):

    def try_get(hrv_dict, key):
        try:
            value = hrv_dict[key]
        except Exception:
            value = np.nan
        return value

    hrv = [
        try_get(hrv_dict, 'HF'),
        try_get(hrv_dict, 'LF'),
        try_get(hrv_dict, 'ULF'),
        try_get(hrv_dict, 'VLF'),
        try_get(hrv_dict, 'VHF'),
        try_get(hrv_dict, 'LFn'),
        try_get(hrv_dict, 'HFn'),
        try_get(hrv_dict, 'LF/P'),
        try_get(hrv_dict, 'HF/P'),
        try_get(hrv_dict, 'sdNN'),
        try_get(hrv_dict, 'cvNN'),
        try_get(hrv_dict, 'CVSD'),
        try_get(hrv_dict, 'LF/HF'),
        try_get(hrv_dict, 'RMSSD'),
        try_get(hrv_dict, 'madNN'),
        try_get(hrv_dict, 'mcvNN'),
        try_get(hrv_dict, 'pNN50'),
        try_get(hrv_dict, 'pNN20'),
        try_get(hrv_dict, 'meanNN'),
        try_get(hrv_dict, 'Triang'),
        try_get(hrv_dict, 'Shannon'),
        try_get(hrv_dict, 'medianNN'),
        try_get(hrv_dict, 'Shannon_h'),
        try_get(hrv_dict, 'Total_Power'),
        try_get(hrv_dict, 'Entropy_SVD'),
        try_get(hrv_dict, 'Fisher_Info'),
        try_get(hrv_dict, 'FD_Petrosian'),
        try_get(hrv_dict, 'Entropy_Spectral_LF'),
        try_get(hrv_dict, 'Entropy_Spectral_HF'),
        try_get(hrv_dict, 'Entropy_Spectral_VLF'),
        try_get(hrv_dict, 'Correlation_Dimension'),
        try_get(hrv_dict, 'Entropy_Multiscale_AUC')
    ]
    return hrv


def ecg_lead_feats(ecg):
    try:
        ecg_dict = ecg_preprocess(
            ecg=ecg,
            sampling_rate=500,
            filter_type='FIR',
            filter_band='bandpass',
            filter_frequency=[3, 45],
            filter_order=0.3,
            segmenter='hamilton'
        )
        ecg_filtered = ecg_dict['df']['ECG_Filtered']

        pqrst = PQRST(ecg_dict)
        time = time_feats(pqrst)  # 12
        amp_origin = amplitude_feats(ecg, pqrst)  # 26
        amp_prep = amplitude_feats(ecg_filtered, pqrst)  # 26
        dist_origin = distance_feats(ecg, pqrst)  # 12
        dist_prep = distance_feats(ecg_filtered, pqrst)  # 12
        slope_origin = slope_feats(ecg, pqrst)  # 10
        slope_prep = slope_feats(ecg_filtered, pqrst)  # 10
        angle_origin = angle_feats(ecg, pqrst)  # 17
        angle_prep = angle_feats(ecg_filtered, pqrst)  # 17

        engs = time + amp_origin + amp_prep + \
            dist_origin + dist_prep + slope_origin + \
            slope_prep + angle_origin + angle_prep
        # 142 in total
    except Exception:
        engs = [np.nan] * 142

    try:
        quality_dict = ecg_signal_quality(
            cardiac_cycles=ecg_dict['ECG']['Cardiac_Cycles'],
            sampling_rate=500,
            rpeaks=ecg_dict['ECG']['R_Peaks']
        )
        quality = quality_feats(quality_dict)  # 10
        # 10 in total
    except Exception:
        quality = [np.nan] * 10

    try:
        hrv_dict = ecg_hrv(
            rpeaks=ecg_dict['ECG']['R_Peaks'],
            sampling_rate=500,
            hrv_features=['time', 'frequency', 'nonlinear']
        )
        hrv = hrv_feats(hrv_dict)  # 32
    except Exception:
        hrv = [np.nan] * 32

    feats = engs + quality + hrv
    return feats


def ecg_feats(ecg12):
    ecg_leads_feats = []
    for i in range(ecg12.shape[0]):
        ecg_leads_feats += ecg_lead_feats(ecg12[i, :])

    ecg_leads_feats = np.array(ecg_leads_feats).reshape((1, -1))
    return ecg_leads_feats


def load_ecg(sample_path):
    with open(sample_path, 'r') as f:
        content = f.readlines()

    content = [list(map(int, c.strip().split())) for c in content[1:]]
    ecg = np.array(content, dtype=np.int16).transpose()

    I, II = ecg[0], ecg[1]
    III = np.expand_dims(II - I, axis=0)
    aVR = np.expand_dims(-(I + II) / 2, axis=0)
    aVL = np.expand_dims(I - II / 2, axis=0)
    aVF = np.expand_dims(II - I / 2, axis=0)
    ecg = np.concatenate([ecg, III, aVR, aVL, aVF], axis=0)
    return ecg


def extract_features(input_dir, output_dir, is_train=False):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    features = None
    samples = os.listdir(input_dir)
    for sample in tqdm(samples, ncols=75):
        sample_path = os.path.join(input_dir, sample)
        ecg = load_ecg(sample_path)
        ecg_leads_feats = ecg_feats(ecg)

        output_file = sample.split('.')[0] + '.npy'
        output_path = os.path.join(output_dir, output_file)
        np.save(output_path, ecg_leads_feats)

        if features is None:
            features = ecg_leads_feats
        else:
            features = np.append(features, ecg_leads_feats, axis=0)

    if is_train:
        features_mean = np.nanmean(features, axis=0)
        features_std = np.nanstd(features, axis=0)
        np.save('./hrv_mean.npy', features_mean)
        np.save('./hrv_std.npy', features_std)
    return


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    extract_features(
        input_dir='../../data/train_txt',
        output_dir='../../data/train_hrv',
        is_train=True
    )

    extract_features(
        input_dir='../../data/testA_txt',
        output_dir='../../data/testA_hrv'
    )
