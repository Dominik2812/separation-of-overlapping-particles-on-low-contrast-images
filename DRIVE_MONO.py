
import pprint
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from PIL import Image, ImageEnhance
import os
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join
import csv
from pathlib import Path



def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % height.round(1),
                ha='center', va='bottom')
                
def make_local_hist(radius_liste):
        
        binning_list1=[(45,55),(56,70),(71,99),(100,124),(125,149),(150,199),(200,249),(250,299),(300,399),(400,419),(420,449),(450,499),(500,600)]#[(int(max_Rad*i/n),int(max_Rad*(i+1)/n)) for i in range(n)]
        binning_list=[(k[0]/2,k[1]/2) for k in binning_list1]
        local_hist=dict()
        for thresh in binning_list:
                local_hist[thresh]=[0] 
                for radius in radius_liste:
                        if thresh[0]<radius<thresh[1]:
                                if local_hist[thresh]==[0]:
                                        print(thresh)
                                        local_hist[thresh]=[radius]                                       
                                else:
                                        local_hist[thresh].append(radius)
        return local_hist

                
def Find_circles(new_path,put,filename,SF1,SF2,KS,C,minDist, pix_um): #21,51,501,(-10->10)
    
    image = cv2.imread(new_path)
    print('OPENED')
    shifted = cv2.pyrMeanShiftFiltering(image, SF1,SF2)#21, 51)



    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                  cv2.THRESH_BINARY,KS,C)#501,10)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=minDist,
            labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    #############################################################################
    kreis_liste1=list()
    radius_liste1=list()
    kreis_DICT=dict()
    for label in np.unique(labels):
            if label == 0:
                    continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            
            if r>minDist:
                kreis_liste1.append((label,(x, y), r))
                kreis_DICT[label]=[(x, y), r]
                radius_liste1.append(r*pix_um)
    print('Radiusliste1',len(radius_liste1),'kreisliste1',len(kreis_liste1))
   
    radius_liste_schlecht=list()
    kreis_liste_schlecht=list()
    schlecht_label=list()
    schlecht_DICT=dict()
    for label1,(x1, y1), r1 in kreis_liste1:
        for label,(x, y), r in kreis_liste1:
            Distanz=((x-x1)**2+(y-y1)**2)**0.5
            if  [(label,(x, y), r),(label1,(x1,y1), r1)] not in kreis_liste_schlecht and [(label1,(x1, y1), r1),(label,(x,y), r)] not in kreis_liste_schlecht and label!=label1:
                if Distanz<0.6*(r+r1):
                    radius_liste_schlecht.append(r)
                    radius_liste_schlecht.append(r1)
                    kreis_liste_schlecht.append([(label,(x, y), r),(label1,(x1,y1), r1)])
                    schlecht_DICT[label]=[(x, y), r]
                    schlecht_DICT[label1]=[(x1, y1), r1]
                   

    radius_liste=list()
    kreis_liste=list()
    kreis_DICT2=dict()
    for label in kreis_DICT:
        if label not in schlecht_DICT:
            print('ich SÄUBERE',label,(x, y), r)
            [(x, y), r]=kreis_DICT[label]
            radius_liste.append(r)
            kreis_liste.append((label,(x, y), r))
            kreis_DICT2[label]=[(x, y), r]
    print('Radiusliste',len(radius_liste),'kreisliste',len(kreis_liste),'kreis_DICT2',len(kreis_DICT2))
    
    kreis_liste3=list()
    kreis_DICT3=dict()
    for label in kreis_DICT2:
        for label1 in kreis_DICT2:
            [(x, y), r]=kreis_DICT2[label]
            [(x1, y1), r1]=kreis_DICT2[label1]
            if label!=label1:
                Distanz=((x-x1)**2+(y-y1)**2)**0.5
                if r+r1>Distanz>0.6*(r+r1):
                    delta=abs(r+r1-Distanz)
                    r=r-r/(r1+r)*delta
                    r1=r1-r1/(r1+r)*delta
                    kreis_DICT2[label]=[(x, y), r]
                    kreis_DICT2[label1]=[(x1, y1), r1]

    kreis_liste3=list()
    radius_liste5=list()
    for label in kreis_DICT2:
        [(x, y), r]=kreis_DICT2[label]
        kreis_liste3.append((label,(x, y), r))
        radius_liste5.append(r*pix_um)
          
                        
            
    
  
    Verdachtsfälle=list()
    for [(label,(x, y), r),(label1,(x1,y1), r1)] in kreis_liste_schlecht:
        if r not in Verdachtsfälle and r1 not in Verdachtsfälle:
            Verdachtsfälle.append(r*pix_um)
            Verdachtsfälle.append(r1*pix_um)


    for label,(x, y), r in kreis_liste3:
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for (label,(x, y), r),(label1,(x1,y1), r1) in kreis_liste_schlecht:
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 10)
        cv2.putText(image, "{}".format(int(r*pix_um)), (int(x) - 10, int(y)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(image, (int(x1), int(y1)), int(r1), (0, 0, 255), 10)
        cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(image, (int(x), int(y)), (int(x1),int(y1)), (0, 0, 255), 3)
        
            
        
    cv2.imwrite(put+'//'+filename, image)
    return  radius_liste5
    


foldername='Monodisperse'
take=r'./'+foldername
Analysiert=r'./'+'Analyzed_Mono'

pixel_to_um_0=250/78
pixel_to_um_1=250/227
pixel_to_um_2=250/412

pixel_to_um_0_16=250/77*4/1.6
pixel_to_um_1_16=250/227*4/1.6
pixel_to_um_1_16_zoom_out=1000/356
pixel_to_um_1_16_zoom_in=1000/380
pixel_to_um_2_16=250/412*4/1.6

onlyfiles = [f for f in listdir(take) if isfile(join(take, f))]
print(onlyfiles)
counter=0
global_radius_liste=list()
global_Verdachts_liste=list()

for file in onlyfiles:
        counter=counter+1
        Bildname=file[0:len(file)-4]
        filename='//'+file
        path=take+filename
        file2=foldername+'_'+Bildname+'_'+'__'+str(counter)+'.JPG'
        radius_liste=Find_circles(path,Analysiert,file2,21,51,501,0,20,pixel_to_um_1_16_zoom_in)


                   
        ####Massenhistogramm, GLOBAL
        for elements in radius_liste:
                global_radius_liste.append(elements)
        
        
        global_hist=make_local_hist(global_radius_liste)
        mass_global=list()
        mass_global_dict=dict()
        for k in global_hist:
            if global_hist[k]!=[0]:
                print(global_hist[k])
                mass_global.append(3.14*4/3*(k[1]-abs(k[0]-k[1])/2)**3*len(global_hist[k]))
                mass_global_dict[k]=3.14*4/3*(k[1]-abs(k[0]-k[1])/2)**3*len(global_hist[k])
            else:
                mass_global.append(0)
                mass_global_dict[k]=0
        
        total_mass=0
        for masse in mass_global:
                total_mass=total_mass+masse
                
        rel_mass_global_dict=dict()
        for k in mass_global_dict:
                rel_mass_global_dict[k]=mass_global_dict[k]/total_mass
                
        rel_mass=[rel_mass_global_dict[k]*100 for k in rel_mass_global_dict]
        Durchmesser=[2*(k[0]) for k in rel_mass_global_dict]
        bin_width=[2*(k[1]-k[0]) for k in rel_mass_global_dict]

        fig, ax = plt.subplots()
        rects1 = ax.bar(Durchmesser, rel_mass, width=bin_width , color='green',align='edge',edgecolor='black')
        autolabel(rects1)

        Path(Analysiert).mkdir(parents=True, exist_ok=True)
        new_path=Analysiert+'//'+foldername+'_'+Bildname+'_'+str(counter)+'__'+'GLOBAL_Histogramm'+'.JPG'
        print(new_path)

        plt.savefig(new_path)
        plt.clf()
        plt.close()

        plt.clf()
        plt.close()

        



Excell_liste=[[2*k[0],rel_mass_global_dict[k]*100] for k in rel_mass_global_dict]
with open(foldername+'_'+Bildname+'__'+str(counter)+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(Excell_liste)
