"""Transparent ECG measurements for bridge evidence; no diagnostic decisions."""
from __future__ import annotations
import math
from typing import Any,Dict,Iterable,List,Optional
import numpy as np
from scipy import signal,stats

LEADS=('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')

def _finite(value):
 try:
  value=float(value);return round(value,6) if math.isfinite(value) else None
 except Exception:return None

def _approx_entropy(x,m=2,r=None):
 x=np.asarray(x,float);n=len(x)
 if n<m+2:return None
 r=float(.2*np.std(x) if r is None else r)
 if r<=0:return 0.
 def phi(mm):
  patterns=np.asarray([x[i:i+mm] for i in range(n-mm+1)]);dist=np.max(np.abs(patterns[:,None]-patterns[None,:]),axis=2);c=np.mean(dist<=r,axis=1);return np.mean(np.log(c+1e-12))
 return float(phi(m)-phi(m+1))

def detect_r_peaks(ecg,sampling_rate):
 """Consensus QRS candidates from lead II and max-energy fallback."""
 x=np.asarray(ecg,float);lead=x[1] if x.shape[0]>1 else x[0];lead=signal.detrend(lead);nyq=sampling_rate/2
 if sampling_rate>=80:
  b,a=signal.butter(2,[5/nyq,min(20/nyq,.95)],btype='band');filtered=signal.filtfilt(b,a,lead)
 else:filtered=lead
 energy=np.convolve(np.gradient(filtered)**2,np.ones(max(1,int(.08*sampling_rate)))/max(1,int(.08*sampling_rate)),mode='same');prom=max(np.percentile(energy,75)*.35,np.std(energy)*.15,1e-12)
 peaks,_=signal.find_peaks(energy,distance=max(1,int(.25*sampling_rate)),prominence=prom)
 return peaks,filtered,energy

def extract_ecg_statistics(ecg,sampling_rate=100,scope='overall',amplitude_units='input_units_unverified'):
 x=np.asarray(ecg,dtype=np.float64)
 if x.ndim!=2:raise ValueError('ECG statistics require a 2-D lead-by-time array')
 if x.shape[0]!=12 and x.shape[1]==12:x=x.T
 if x.shape[0]!=12:raise ValueError(f'Expected 12 leads, got {x.shape}')
 if not np.isfinite(x).all():x=np.nan_to_num(x)
 duration=x.shape[1]/float(sampling_rate);peaks,filtered,energy=detect_r_peaks(x,sampling_rate);rr=np.diff(peaks)/sampling_rate*1000.;valid_rr=rr[(rr>=300)&(rr<=2000)];diff=np.diff(valid_rr)
 hr=60000/valid_rr if len(valid_rr) else np.array([]);amp=x[1,peaks] if len(peaks) else np.array([])
 hist_count=0
 if len(valid_rr):hist_count=int(np.histogram(valid_rr,bins=max(1,int(np.ptp(valid_rr)/7.8125)+1))[0].max())
 qrs=[]
 for peak in peaks:
  threshold=energy[peak]*.12;left=peak;right=peak
  limit=int(.12*sampling_rate)
  while left>max(0,peak-limit) and energy[left]>threshold:left-=1
  while right<min(len(energy)-1,peak+limit) and energy[right]>threshold:right+=1
  width=(right-left)/sampling_rate*1000
  if 35<=width<=240:qrs.append(width)
 lead_stats={}
 for name,row in zip(LEADS,x):
  lead_stats[name]={'mean_amplitude':_finite(np.mean(row)),'std_amplitude':_finite(np.std(row)),'rms_amplitude':_finite(np.sqrt(np.mean(row**2))),'peak_to_peak_amplitude':_finite(np.ptp(row)),'skewness':_finite(stats.skew(row)),'kurtosis':_finite(stats.kurtosis(row))}
 short=duration<60;enough=len(valid_rr)>=3
 time_domain={'mean_rr_ms':_finite(np.mean(valid_rr)) if enough else None,'median_rr_ms':_finite(np.median(valid_rr)) if enough else None,'sdnn_ms':_finite(np.std(valid_rr,ddof=1)) if len(valid_rr)>=4 else None,'rmssd_ms':_finite(np.sqrt(np.mean(diff**2))) if len(diff)>=2 else None,'pnn20_percent':_finite(100*np.mean(np.abs(diff)>20)) if len(diff)>=2 else None,'pnn50_percent':_finite(100*np.mean(np.abs(diff)>50)) if len(diff)>=2 else None,'rr_cv':_finite(np.std(valid_rr,ddof=1)/np.mean(valid_rr)) if len(valid_rr)>=4 else None,'hrv_triangular_index':_finite(len(valid_rr)/hist_count) if hist_count else None,'approximate_entropy':_finite(_approx_entropy(valid_rr)) if len(valid_rr)>=12 else None}
 return {'scope':scope,'duration_seconds':_finite(duration),'sampling_rate_hz':int(sampling_rate),'amplitude_units':amplitude_units,'measurement_status':'measured_with_limitations','limitations':(['HRV values are short-window descriptors, not standard long-term clinical HRV.'] if short else [])+(['Too few valid RR intervals for some HRV metrics.'] if not enough else [])+(['Amplitude units were not physically verified; amplitude statistics are relative only.'] if amplitude_units=='input_units_unverified' else [])+['R peaks and QRS widths are algorithmic estimates and require clinical waveform validation.','PR, QT/QTc and P/T amplitudes are unavailable without a validated wave-delineation model.'],'beat_detection':{'r_peak_count':int(len(peaks)),'valid_rr_count':int(len(valid_rr)),'mean_heart_rate_bpm':_finite(np.mean(hr)) if len(hr) else None,'median_heart_rate_bpm':_finite(np.median(hr)) if len(hr) else None},'time_domain_hrv':time_domain,'morphology':{'qrs_duration_median_ms_estimate':_finite(np.median(qrs)) if qrs else None,'qrs_duration_iqr_ms_estimate':_finite(np.subtract(*np.percentile(qrs,[75,25]))) if len(qrs)>=4 else None,'lead_ii_r_amplitude_median':_finite(np.median(amp)) if len(amp) else None,'lead_ii_r_amplitude_iqr':_finite(np.subtract(*np.percentile(amp,[75,25]))) if len(amp)>=4 else None,'pr_interval_ms':None,'qt_interval_ms':None,'qtc_bazett_ms':None,'p_wave_amplitude':None,'t_wave_amplitude':None},'per_lead_statistics':lead_stats}

def build_segment_statistics(overall_ecg,sampling_rate=100,abnormal_segments=None):
 segments=[]
 for i,item in enumerate(abnormal_segments or []):
  waveform=item.get('ecg') if isinstance(item,dict) else item;meta=item if isinstance(item,dict) else {}
  measured=extract_ecg_statistics(waveform,sampling_rate,scope='abnormal_window');measured['window_index']=meta.get('window_index',i);measured['start_seconds']=meta.get('start_seconds');measured['end_seconds']=meta.get('end_seconds');measured['selection_probability']=meta.get('abnormal_probability');segments.append(measured)
 return {'overall':extract_ecg_statistics(overall_ecg,sampling_rate,scope='overall'),'abnormal_windows':segments,'comparison':compare_scopes(extract_ecg_statistics(overall_ecg,sampling_rate,scope='overall'),segments)}

def compare_scopes(overall,segments):
 if not segments:return {'status':'unavailable','reason':'No abnormal windows supplied'}
 keys=(('beat_detection','mean_heart_rate_bpm'),('time_domain_hrv','mean_rr_ms'),('time_domain_hrv','rmssd_ms'),('morphology','qrs_duration_median_ms_estimate'),('morphology','lead_ii_r_amplitude_median'));result={}
 for group,key in keys:
  base=overall[group].get(key);vals=[x[group].get(key) for x in segments if x[group].get(key) is not None];mean=_finite(np.mean(vals)) if vals else None;result[key]={'overall':base,'abnormal_mean':mean,'difference':_finite(mean-base) if mean is not None and base is not None else None}
 return {'status':'available','metrics':result}

def allocate_retrieval_queries(abnormal_windows,overall_windows,abnormal_count=3,overall_count=2):
 """Deterministic 3:2 selection; returns query descriptors, not FAISS results."""
 abnormal=sorted(abnormal_windows or [],key=lambda x:float(x.get('abnormal_probability',0)),reverse=True)[:abnormal_count];used={x.get('window_index') for x in abnormal};overall=[x for x in (overall_windows or []) if x.get('window_index') not in used];overall=sorted(overall,key=lambda x:float(x.get('representativeness',0)),reverse=True)[:overall_count]
 return {'ratio':'3:2','abnormal_queries':abnormal,'overall_reference_queries':overall,'requested':{'abnormal':abnormal_count,'overall_reference':overall_count},'selected':{'abnormal':len(abnormal),'overall_reference':len(overall)}}
