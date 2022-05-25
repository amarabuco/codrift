
from river import drift

def get_drifts(data, col, detector='adwin'):
    if (detector == 'adwin'):
        drift_detector = drift.ADWIN(delta=0.001)
    
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