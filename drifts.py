
from river import drift
import pandas as pd

def get_drifts(data, col, detector='adwin'):
    if (detector == 'adwin'):
        drift_detector = drift.ADWIN(delta=0.001)
    if (detector == 'kswin'):
        drift_detector = drift.KSWIN(window_size=100, stat_size=30, alpha=0.005, seed=42)
    if (detector == 'ddm'): #stationarity 
        drift_detector = drift.DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)
    if (detector == 'eddm'): #stationarity 
        drift_detector = drift.EDDM(min_num_instances=30, warning_level=0.95, out_control_level=0.9)
    if (detector == 'hddma'): #stationarity 
        drift_detector = drift.HDDM_A(drift_confidence=0.001, warning_confidence=0.005, two_sided_test=False)
    if (detector == 'hddmw'): #stationarity 
        drift_detector = drift.HDDM_W(drift_confidence=0.001, warning_confidence=0.005, lambda_option=0.05, two_sided_test=False)
    if (detector == 'page'):  
        drift_detector = drift.PageHinkley(min_instances=6)

    data_diff = data.to_dict()[col]
    drifts =[]
    drift_data = {}
   
    for k in data_diff:
        #print(k)
        in_drift, in_warning = drift_detector.update(data_diff[k])
        if in_drift:
            drifts.append(k) 
            #print(f"Change detected at index {k}, input value: {data[k]}")  
    for key, drift_point in enumerate(drifts):
        if key == 0:
            drift_data[key] = data[:drift_point]
        elif key == len(drifts)-1:
            drift_data[key] = data[drifts[key-1]:drift_point]
            drift_data[key+1] = data[drift_point:]
        else:
            drift_data[key] = data[drifts[key-1]:drift_point]
    
    return drifts, drift_data

def set_detector(detector, **kwargs):
    if (detector == 'adwin'):
        #drift_detector = drift.ADWIN(delta=0.001)
        drift_detector = drift.ADWIN(**kwargs)
    if (detector == 'kswin'):
        #drift_detector = drift.KSWIN(window_size=60, stat_size=14, alpha=0.005)
        drift_detector = drift.KSWIN(**kwargs)
    if (detector == 'ddm'): #error 
        #drift_detector = drift.DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)
        drift_detector = drift.DDM()
    if (detector == 'eddm'): #error 
        drift_detector = drift.EDDM()
        #drift_detector = drift.EDDM(min_num_instances=30, warning_level=0.95, out_control_level=0.9)
    if (detector == 'hddma'): #infinitos drifts
        #drift_detector = drift.HDDM_A(drift_confidence=0.001, warning_confidence=0.005, two_sided_test=True)
        drift_detector = drift.HDDM_A(**kwargs)
    if (detector == 'hddmw'): #infinitos drifts 
        #drift_detector = drift.HDDM_W(drift_confidence=0.001, warning_confidence=0.005, lambda_option=0.05, two_sided_test=True)
        drift_detector = drift.HDDM_W(**kwargs)
    if (detector == 'page'):  
        #drift_detector = drift.PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
        #drift_detector = drift.PageHinkley(**kwargs)
        drift_detector = drift.PageHinkley(min_instances=6)
    return drift_detector
    
def get_drifts2(data: pd.Series, col: int, drift_detector):
    data_diff = data.to_dict()[col]
    drifts =[]
    drift_data = {}
   
    for k in data_diff:
        #print(k)
        in_drift, in_warning = drift_detector.update(data_diff[k])
        if in_drift:
            drifts.append(k) 
            #print(f"Change detected at index {k}, input value: {data[k]}")  
    for key, drift_point in enumerate(drifts):
        if key == 0:
            drift_data[key] = data[:drift_point]
        elif key == len(drifts)-1:
            drift_data[key] = data[drifts[key-1]:drift_point]
            drift_data[key+1] = data[drift_point:]
        else:
            drift_data[key] = data[drifts[key-1]:drift_point]
    params = drift_detector._get_params()
    if ('window' in params):
        params.pop('window')

    return drifts, drift_data, params

def drifts2df(drift_data):
    dfts = pd.DataFrame()
    for k, d in drift_data.items():
        tmp = pd.DataFrame(d)
        tmp['drift'] = k
        dfts = dfts.append(tmp)
    return dfts