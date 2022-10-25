#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:38:38 2022

@author: hannekevandijk
"""


def main_NCG20_QDS():
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
    from tqdm.auto import tqdm

    from seaborn import heatmap
    from matplotlib.backends.backend_pdf import PdfPages

    from ncg_code.inout import FilepathFinder as ff
    ignore = ['._','DS','.ipynb']
    #startdir = 'data/'
    startdir = '/home/data/'
    dirs = os.listdir(startdir+'input/')

    dirs = [d for d in dirs if not 'DS' in d and not any([ig in d for ig in ignore])]
    #print(dirs)
    for d in dirs:
        ecg = ff('.txt',startdir+'input/'+d)
        ecg.get_filenames()
        ecg.files = np.sort(ecg.files)
        
        files = [f for f in ecg.files if not '._' in f and not any([ig in f for ig in ignore])]
        files = np.sort(files)
        path,tail = os.path.split(files[0])
        filen, ext = os.path.splitext(tail)

        with PdfPages(startdir+'output/'+d+'.pdf') as pp:
            nr=1
            BHmarker=np.zeros((16));slopes=[];rvalues=[];pvalues=[]
            for f in tqdm(files):
                fpath, ext = os.path.splitext(f)
                head,tail = os.path.split(fpath)
                startdata = np.array(pd.read_csv(f,sep=',').iloc[:,0])
                sF = 130

                onecycle = (5+11) # in seconds
                steps = 15 #15+1 rest
                totalstimlength = (onecycle*(steps+1)*sF)
                end = totalstimlength
                start = 0
                trainstart = np.arange(0,(onecycle*(steps+2)*sF),onecycle*sF)

                data = startdata[int(start):int(end)]

                del startdata

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
                # annotmarker = np.zeros((len(fdata)))
                # annotmarker[rpeaks2]=1

                #for the loop to work we add interpolation of HR at end (because of the diff computation in rpeaks2)
                HR = np.hstack((HR,HR[-1]+(HR[-1]-HR[-2])))
                meanHR = np.zeros((len(fdata)));meanHR[:]=np.nan
                i = 0
                for h in range(len(fdata)):
                    if i == len(rpeaks2):
                        meanHR[h]=HR[i-1]
                    else:
                        meanHR[h]=np.mean(HR[i:i+1])
                        if h == rpeaks2[i]:
                            i = i+1

                paddedHR = np.pad(meanHR[~np.isnan(meanHR)],2*sF, mode='reflect') #zeropad the data to be able to use more wavelets
                hanndata = convolve(paddedHR, hann(int(2*sF)), mode ='same', method ='auto')/sum(hann(int(2*sF)))
                hanndata = hanndata[2*sF:]
                hanndata = hanndata[:-2*sF]

                prepadding=[];postpadding=[]
                for p in range(50):
                    if p%2==0:
                        prepadding = np.hstack((prepadding, hanndata[:trainstart[1]][::-1]))
                        postpadding = np.hstack((postpadding, hanndata[trainstart[-2]:][::-1]))
                    else:
                        prepadding = np.hstack((prepadding, hanndata[:trainstart[1]]))
                        postpadding = np.hstack((postpadding, hanndata[trainstart[-2]:]))

                paddeddata = np.hstack((prepadding,hanndata,postpadding))#np.pad(hanndata,padlen, mode='wrap')#mode='reflect') #zeropad the data to be able to use more wavelets

                """ specific for reports """
                fortfrdata = np.expand_dims(np.expand_dims(paddeddata,axis=0),axis=0)#mne requires a matrix with dimensions: segments x channels x samples

                freqs=np.round(np.arange(0.02,0.18,0.0005),decimals=4)

                tfr3 = mne.time_frequency.tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=3)#maxperlen)
                tfr3 = tfr3[0,0,:,len(prepadding):]
                tfr3 = tfr3[:,:-len(postpadding)]
                tfr3 = (tfr3-np.min(tfr3))/(np.max(tfr3)-np.min(tfr3))*10

                tfr10 = mne.time_frequency.tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=10)#maxperlen)
                tfr10 = tfr10[0,0,:,len(prepadding):]
                tfr10 = tfr10[:,:-len(postpadding)]

                meanTFR = np.array(tfr10[np.where(freqs==0.0625)[0][0],:])
                meanTFR = (meanTFR-np.min(meanTFR))/(np.max(meanTFR)-np.min(meanTFR))*10

                tfr10 = (tfr10-np.min(tfr10))/(np.max(tfr10)-np.min(tfr10))*10

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
                #print(f)

                # slope, intercept, rvalue, pvalue, _= linregress(np.arange(0,len(HBmarker)),HBmarker)
                # slopes.append(slope)
                # rvalues.append(rvalue**2)
                # pvalues.append(pvalue)

                fig,axs=plt.subplots(6,1,figsize=(8.23,11.69))
                fig.suptitle(str(nr)+': '+tail, fontsize=10)

                axs[0].plot(fdata)
                #axs[0].plot(marker,fdata[marker], 'g', marker = 7)
                axs[0].set_title('ECG')
                axs[0].set_xlim(0,len(fdata))
                axs[0].plot(annotdata,marker=3,linewidth=.001,label='R peaks')
                axs[0].set_ylabel('ECG')
                #axs[0].set_ylim(np.nanmean(fdata)-5*np.nanstd(fdata),np.nanmean(fdata)+10*np.nanstd(fdata))
                axs[0].vlines(trainstart,axs[0].get_ylim()[0],axs[0].get_ylim()[1],'k', linewidth = .5)
                axs[0].set_xticklabels([])
                axs[0].set_xticks([])

                axs[1].set_title('mean HR')
                axs[1].plot(hanndata)
                axs[1].set_ylabel('mean HR (BPM)')
                axs[1].set_ylim(np.nanmean(hanndata)-5*np.nanstd(hanndata),np.nanmean(hanndata)+5*np.nanstd(hanndata))
                axs[1].vlines(trainstart, axs[1].get_ylim()[0], axs[1].get_ylim()[1], 'k', linewidth =  .5)
                axs[1].set_xlim(0,len(fdata))
                axs[1].set_xticklabels([])
                axs[1].set_xticks([])


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
                axs[2].set_xticks([])
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
                axs[3].set_xticks([])
                axs[3].set_ylim(0,len(freqs))

                axs[4].set_title('meanTFR @ 0.0625Hz')
                axs[4].plot(meanTFR)
                axs[4].set_ylabel('(uV2)')
                axs[4].set_ylim(0,10)
                axs[4].set_xlim(0,len(fdata))
                axs[4].set_xticklabels([])
                axs[4].set_xticks([])
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

            BHmarker=BHmarker[1:]
            scaledBHmarker = np.asarray((BHmarker-np.min(BHmarker))/(np.max(BHmarker)-np.min(BHmarker)))#BHmarker[1:])/10
            locations=np.array(np.arange(1,scaledBHmarker.shape[0]+1),dtype=str)
            intensities = np.array(np.arange(0,scaledBHmarker.shape[1]),dtype=str)

            scaledpower = np.round(np.mean(scaledBHmarker[:,3:],axis=1),decimals=2)#(powers-np.nanmin(powers))/(np.nanmax(powers)-np.nanmin(powers)),decimals=2)
            scslopes=[];scrvalues=[];scpvalues=[]
            for s in range(scaledBHmarker.shape[0]):
                scslope, scintercept, scrvalue, scpvalue, _= linregress(np.arange(0,len(scaledBHmarker[s])),scaledBHmarker[s])
                scslopes.append(scslope)
                scrvalues.append(scrvalue**2)
                scpvalues.append(np.round(scpvalue,decimals=2))
            #keys.append(nlocs[l])
            scaledBHmarker = np.round(scaledBHmarker,decimals=2)

            df = pd.DataFrame(scaledBHmarker.T, columns=locations, index=intensities)
            #        df = df.drop(0,axis=0)
            heatmap(df.transpose(),ax=axes[0],annot=True,square=False,cmap='Oranges', cbar=False, vmin=0, vmax=1)
            axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation = 0, fontsize = 8)
            #axes[0].set_title('Score')
            #a=a+1
            axes[0].set_xlabel('TMS pulse intensity')

            smarker = np.zeros((len(scpvalues)))
            smarker[np.array(scpvalues)<0.05]=1
            #determining maximum power
            pmarker = np.zeros((len(scaledpower)))
            pmarker[np.argmax(scaledpower)]=1

            dfstats = pd.DataFrame([scaledpower, np.array(scslopes), np.array(scrvalues),np.array(scpvalues)], columns=locations,index=['pow','slope','r','p'])

            maxp = np.argmax(scaledpower)
            maxr = np.argmax(scrvalues)
            sigslopes = np.where(np.array(scpvalues)<0.05)[0]
            if len(sigslopes)>0 and any(np.array(scrvalues)[sigslopes]>0):#only positive slopes
                for marker in sigslopes[np.array(scrvalues)[sigslopes]>0]:
                    axes[0].add_patch(Rectangle((0,marker), 0, 1, edgecolor='lightskyblue', fill=False, lw=10))
                    axes[0].margins(x=0, y=0)
                    axes[0].get_yticklabels()[marker].set_weight('bold')
                    axes[0].get_yticklabels()[marker].set_size(12)
                    axes[0].get_yticklabels()[marker].set_color('lightskyblue')
            axes[0].add_patch(Rectangle((0,maxp), 0, 1, edgecolor='dodgerblue', fill=False, lw=10))
            axes[0].margins(x=0, y=0)
            axes[0].get_yticklabels()[maxp].set_weight('bold')
            axes[0].get_yticklabels()[maxp].set_size(12)
            axes[0].get_yticklabels()[maxp].set_color('dodgerblue')

            df2 = pd.concat([df,dfstats])
            df2.to_csv(startdir+'output/'+d+'_HBstats.csv')
            heatmap(dfstats.transpose(),ax=axes[1],annot=True,cmap = 'Greens',cbar=False, annot_kws={"size": 6})

            axes[1].set_title('Statistics')
            pp.savefig(dpi=1200,transparent=False)
            plt.close()


if __name__ == '__main__':
    main_NCG20_QDS()
