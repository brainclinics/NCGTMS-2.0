#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:38:38 2022

@author: hannekevandijk
"""


def main_NCG200():
    import sys
    import os
    import numpy as np
    import mne
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import find_peaks
    from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, hilbert, convolve, boxcar, medfilt
    from scipy.signal.windows import hann
    from scipy.stats import zscore
    from matplotlib.patches import Rectangle
    from scipy.stats import linregress
    import datetime
    
    from seaborn import heatmap
    from matplotlib.backends.backend_pdf import PdfPages
    
    from ncg_code.inout import FilepathFinder as ff
    
    dirs = os.listdir('data/input/')
    dirs = [d for d in dirs if not 'DS' in d and not '._' in d]
    for d in dirs:
        ecg = ff('.txt', 'data/input/'+d)
        ecg.get_filenames()
        ecg.files = np.sort(ecg.files)
    
        files = [f for f in ecg.files if not '._' in f]
        files = np.sort(files)
        path,tail = os.path.split(d)
        filen, ext = os.path.splitext(tail)
    
        with PdfPages('data/output/'+d+'.pdf') as pp:
            nr=0
            BHmarker=np.zeros((1,15));slopes=[];rvalues=[];pvalues=[]
            for f in files:
                fpath, ext = os.path.splitext(f)
                head,tail = os.path.split(fpath)
                data = np.array(pd.read_csv(f,sep=',').iloc[:,0])
                sF = 128
         
                highpass = 5 #/ nyq
                lowpass = 63.9999
                # ''' bandpassfilter '''
                sos = butter(4,[highpass,lowpass], btype='bandpass', analog=False, output = 'sos', fs=sF)
                fdata = sosfiltfilt(sos, data)
                
                marker =  np.where(zscore(fdata)>8.5)[0][0] #QDS rest period before starting first stim       
                onecycle = (5+11) # in seconds
                steps = 15 #15+1 rest
                end = marker+(onecycle*steps*sF)
                if marker-(16*sF)<0:
                    start = 0
                else:
                    start = marker-(16*sF)
    
                
                data = data[int(start):int(end)]
                trainstart = np.arange(0,(onecycle*steps*sF)+onecycle,onecycle*sF)
    
            
                highpass = 5 #/ nyq
                lowpass = 49
                # ''' bandpassfilter '''
                sos = butter(4,[highpass,lowpass], btype='bandpass', analog=False, output = 'sos', fs=sF)
                fdata = sosfiltfilt(sos, data)
                        
                peaks,amps = find_peaks(fdata, prominence=4*np.std(fdata)+np.mean(fdata), wlen=.25*sF)
                annotdata = np.zeros((len(fdata))); annotdata[:] = np.nan
                #only take r-peaks that fall within 2SD of the mean peak-hight
                peakz = zscore(fdata[peaks])#amps['peak_heights'])
                rpeaks = peaks[np.where(np.logical_and(peakz>=-7,peakz<7))]#TO DO make about SD not zscores?
                annotdata[rpeaks]=fdata[peaks][np.where(np.logical_and(peakz>=-7,peakz<7))]
                
                RRinterval = np.diff(rpeaks)/sF
                zRR = zscore(RRinterval)
                toofast= np.where(zRR<-5)[0]
                rpeaks2 =rpeaks.copy()
                RRinterval2 = RRinterval.copy()
                if len(toofast)>0:
                    skip = np.where(np.diff(toofast)==1)[0]
                    if len(skip)>0:
                        toofast = np.delete(toofast,skip+1)
                    rpeaks2 = np.delete(rpeaks2,toofast+1)
    
                    newRpeaks = annotdata.copy()
                    newRpeaks[:]=np.nan
                    annotdata[rpeaks2]=fdata[rpeaks2]
                        
                    RRinterval2 = np.diff(rpeaks2)/sF
    
                HR = 60/RRinterval2
                
                marker = np.zeros((len(fdata)))
                marker[rpeaks2]=1
                
                meanHR = np.zeros((len(fdata)));meanHR[:]=np.nan
                i = 0
                for h in range(len(fdata)):
                    if i>len(HR)-1:
                         meanHR[h:]=np.mean(HR[i-6:i-1])
                         break
                    meanHR[h]=np.mean(HR[i:i+5])
                    if marker[h] == 1:
                         i = i+1
                
                paddedHR = np.pad(meanHR[~np.isnan(meanHR)],2*sF, mode='reflect') #zeropad the data to be able to use more wavelets
                hanndata = convolve(paddedHR, hann(int(2*sF)), mode ='same', method ='auto')/sum(hann(int(2*sF)))
                hanndata = hanndata[2*sF:]
                hanndata = hanndata[:-2*sF]
                padlen = 360*sF
                tmpdata = hanndata[~np.isnan(hanndata)] #remove the nan values (caused by hanning smoothing)
                paddeddata = np.pad(tmpdata,padlen, mode='edge')#mode='reflect') #zeropad the data to be able to use more wavelets
                fortfrdata = np.expand_dims(np.expand_dims(paddeddata,axis=0),axis=0)#mne requires a matrix with dimensions: segments x channels x samples
                
                freqs=np.round(np.arange(0.02,0.18,0.0005),decimals=4)
                
                tfr3 = mne.time_frequency.tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=3)#maxperlen)
                tfr3 = tfr3[0,0,:,padlen:]
                tfr3 = tfr3[:,:-padlen]
                #    tfr3=tfr3[:,:xcoords[-1]]
                tfr3 = (tfr3-np.min(tfr3))/(np.max(tfr3)-np.min(tfr3))*10
                
                tfr10 = mne.time_frequency.tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=10)#maxperlen)
                tfr10 = tfr10[0,0,:,padlen:]
                tfr10 = tfr10[:,:-padlen]
                #    tfr10 = tfr10[:,:xcoords[-1]]
                tfr10 = (tfr10-np.min(tfr10))/(np.max(tfr10)-np.min(tfr10))*10
                
                meanTFR = np.array(tfr10[np.where(freqs==0.0625)[0][0],:])
                               
                score = []
                a = 0;
                for x in trainstart[1:]:
                    #print([a,x])
                    if a>0 and a == x:
                        break
                    score.append(np.mean(meanTFR[a:x]))
                    a=x
                score = np.array(score)
                score = score[~np.isnan(np.array(score))]
            
                HBmarker = []
                a=0
                for x in trainstart[1:]:
                    if a>0 and a == x:
                        break
                    HBmarker.append(np.mean(meanTFR[a:x]))
                    a=x
                HBmarker = np.array(HBmarker)
            
                plotscore = np.zeros((trainstart[-1]))
                a = 0;b=0
                for x in trainstart[1:]:
                    if a>0 and a == x:
                        break
                    plotscore[a:x] = HBmarker[b]
                    a=x;b=b+1
                
                BHmarker = np.vstack((BHmarker,HBmarker))
                print(f)
                
                slope, intercept, rvalue, pvalue, _= linregress(np.arange(0,len(HBmarker)),HBmarker)
                slopes.append(slope)
                rvalues.append(rvalue**2)
                pvalues.append(pvalue)
        
                fig,axs=plt.subplots(6,1,figsize=(8.23,11.69))
                fig.suptitle(str(nr)+': '+tail, fontsize=10)
                
                axs[0].plot(fdata)
                axs[0].set_title('ECG') 
                axs[0].set_xlim(0,len(fdata))
                axs[0].plot(annotdata,'*',label='R peaks')
                axs[0].set_ylabel('ECG')
                axs[0].set_ylim(np.nanmean(fdata)-5*np.nanstd(fdata),np.nanmean(fdata)+10*np.nanstd(fdata))
                axs[0].vlines(trainstart,axs[0].get_ylim()[0],axs[0].get_ylim()[1],'k', linewidth = .5)
                axs[0].set_xticklabels([])
                   
                axs[1].set_title('mean HR')                
                axs[1].plot(hanndata)
                axs[1].set_ylabel('mean HR (BPM)')
                axs[1].set_ylim(np.nanmean(hanndata)-5*np.nanstd(hanndata),np.nanmean(hanndata)+5*np.nanstd(hanndata))            
                axs[1].vlines(trainstart, axs[1].get_ylim()[0], axs[1].get_ylim()[1], 'k', linewidth =  .5)
                axs[1].set_xlim(0,len(fdata))
                axs[1].set_xticklabels([])
                
                
                axs[2].set_title('TFR high time resolution')
                tfr3mean = np.nanmean(tfr3[np.where(freqs==0.0625),:])
                tfr3std = 1.5*np.nanstd(tfr3[np.where(freqs==0.0625),:])
                im3 = axs[2].imshow(tfr3[:,:], vmin=tfr3mean-tfr3std, vmax=tfr3mean+tfr3std, cmap = 'coolwarm', aspect='auto')
                im3 = axs[2].plot(np.ones((tfr3.shape[1]))*np.where(freqs==0.0625)[0],'g', alpha = 0.8)
                axs[2].vlines(trainstart, axs[2].get_ylim()[0], axs[2].get_ylim()[1], 'k', linewidth =  .5)
                axs[2].set_ylabel('Frequency (Hz)')
                axs[2].set_yticks([0, len(freqs)/2, len(freqs)])
                axs[2].set_yticklabels(['0.02', '0.1', '0.18'])
                axs[2].set_ylim(0,len(freqs))
                axs[2].set_xticklabels([])
                axs[2].set_xlim(0,len(fdata))
        
                
                axs[3].set_title('TFR high frequency resolution')
                tfr10mean = np.nanmean(tfr10[np.where(freqs==0.0625),:])
                tfr10std = 1.5*np.nanstd(tfr10[np.where(freqs==0.0625),:])
                im4 = axs[3].imshow(tfr10[:,:], vmin=tfr10mean-tfr10std, vmax=tfr10mean+tfr10std, cmap = 'coolwarm', aspect='auto')
                im4 = axs[3].plot(np.ones((tfr10.shape[1]))*np.where(freqs==0.0625)[0],'g', alpha = 0.8)
                axs[3].vlines(trainstart, axs[3].get_ylim()[0], axs[3].get_ylim()[1], 'k', linewidth =  .5)
                axs[3].set_yticks([0, len(freqs)/2, len(freqs)])
                axs[3].set_yticklabels(['0.02', '0.1', '0.18'])
                axs[3].set_xlim(0,len(fdata))
                axs[3].set_xticklabels([])
                axs[3].set_ylim(0,len(freqs))
                
                axs[4].set_title('meanTFR @ 0.0625Hz')
                axs[4].plot(meanTFR)
                axs[4].set_ylabel('(uV2)')
                axs[4].set_ylim(0,10)
                axs[4].set_xlim(0,len(fdata))
                axs[4].set_xticklabels([])
                axs[4].vlines(trainstart,axs[4].get_ylim()[0],axs[4].get_ylim()[1], 'k', linewidth =  .5)
                    
                axs[5].set_title('meanTFR/Intensity @ 0.0625Hz')
                axs[5].plot(plotscore)
                axs[5].fill_between(np.arange(0,len(plotscore)),plotscore, alpha = 0.5)
                axs[5].set_ylabel('(uV2)')
                axs[5].set_xticks(trainstart)
                axs[5].set_ylim(0,10)        
                axs[5].vlines(trainstart,axs[5].get_ylim()[0],axs[5].get_ylim()[1], 'k', linewidth =  .5)
                axs[5].set_xlim(0,trainstart[-1])
                axs[5].set_xticklabels(np.arange(0,len(trainstart)))
            
                pp.savefig()#dpi=1200,transparent=False)
                plt.close()
                nr = nr+1
    
    
            fig, axes = plt.subplots(1,2,sharey=True,gridspec_kw={'width_ratios': [16, 3]},figsize=(8.23,4))
            fig.suptitle('Brain-Heart marker score')
            fig.set_tight_layout(True)
            
            scaledBHmarker = np.asarray(BHmarker[1:])/10#(BHmarker-np.min(BHmarker))/(np.max(BHmarker)-np.min(BHmarker))
            scaledBHmarker = np.round(scaledBHmarker,decimals=2)
            locations=np.array(np.arange(0,scaledBHmarker.shape[0]),dtype=str)
            intensities = np.array(np.arange(0,scaledBHmarker.shape[1]),dtype=str)
            
            df = pd.DataFrame(np.array(scaledBHmarker).T, columns=locations, index=intensities)
            #        df = df.drop(0,axis=0)
            heatmap(df.transpose(),ax=axes[0],annot=True,square=False,cmap='coolwarm', cbar=False, vmin=0, vmax=1)
            axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation = 0, fontsize = 8)
            #axes[0].set_title('Score')
            #a=a+1
            axes[0].set_xlabel('TMS pulse intensity')
            
            sigslopes = np.where(np.array(pvalues)<0.05)[0]
            if len(sigslopes)>0:
                for marker in sigslopes:
                    if np.array(slopes)[marker]>0: #only positive slopes
                        axes[0].add_patch(Rectangle((0,marker), 0, 1 , edgecolor='magenta', fill=False, lw=8))
            
            locations=np.array(np.arange(0,len(scaledBHmarker)),dtype=str)
            
            stats = np.round(np.array([np.mean(scaledBHmarker, axis=1),slopes,rvalues,pvalues]).T, decimals=2)
            dfstats = pd.DataFrame(stats.T,columns=locations, index=['pow','slope','r','p'])       
            df2 =pd.concat([df,dfstats])
            df2.transpose().to_csv('data/output/'+d+'_HBstats.csv')
            heatmap(dfstats.transpose(),ax=axes[1],annot=True,cmap = 'Greens',cbar=False)
            axes[1].set_title('Statistics')
            pp.savefig(dpi=1200,transparent=False)
            plt.close()           


if __name__ == '__main__':
    main_NCG200()

