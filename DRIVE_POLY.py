import pprint
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from PIL import Image, ImageEnhance
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

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def Find_circles(new_path,put,filename,SF1,SF2,KS,C,minDist, pix_um): #21,51,501,(-10->10)
    
    image = cv2.imread(new_path)
    print('OPENED')
    shifted = cv2.pyrMeanShiftFiltering(image, SF1,SF2)#21, 51)
    #cv2.imshow("Input", shifted)


    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                  cv2.THRESH_BINARY,KS,C)#501,10)

    D = ndimage.distance_transform_edt(thresh)
    #cv2.imshow("Thresh", D)
    localMax = peak_local_max(D, indices=False, min_distance=minDist,
            labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    #############################################################################
    kreis_liste=list()
    radius_liste=list()
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
                kreis_liste.append(((x, y), r))
                radius_liste.append(r*pix_um)
    
    radius_liste_schlecht=list()
    kreis_liste_schlecht=list()
    checked=list()
    for (x1, y1), r1 in kreis_liste:
        for (x, y), r in kreis_liste:
            Distanz=((x-x1)**2+(y-y1)**2)**0.5
            if  [((x, y), r),((x1,y1), r1)] not in kreis_liste_schlecht and [((x1, y1), r1),((x,y), r)] not in kreis_liste_schlecht and(x1,y1)!=(x,y):
                if Distanz<0.8*(r+r1):
                    radius_liste_schlecht.append(r)
                    kreis_liste_schlecht.append([((x, y), r),((x1,y1), r1)])

                    


    Verdachtsfälle=list()
    for [((x, y), r),((x1,y1), r1)] in kreis_liste_schlecht:
        if r*pix_um not in Verdachtsfälle and r1*pix_um not in Verdachtsfälle:
            Verdachtsfälle.append(r*pix_um)
            Verdachtsfälle.append(r1*pix_um)
    print('schlecht',len(kreis_liste_schlecht), len(Verdachtsfälle))

    radius_liste2=[r_pix_um for r_pix_um in radius_liste if r_pix_um not in Verdachtsfälle]

    kreis_liste2=[((x, y), r) for ((x, y), r) in kreis_liste if r*pix_um not in Verdachtsfälle]
    print('radius, radius2,Verdachtsfälle',len(radius_liste), len(radius_liste2), len(Verdachtsfälle))






 
    global_hist=make_local_hist(radius_liste2)

                        

        
    FG=0
    HG=300
    GG=400
    FG0=0
    HG0=0
    GG0=0
    for k in global_hist:
        print('k=',k)
        if 2*k[0]<HG:
            FG0=FG0+global_hist[k]
    for k in global_hist:
        print('k=',k)
        if HG <= 2*k[0] <= GG:
            print(True)
            HG0=HG0+global_hist[k]
    for k in global_hist:
        print('k=',k)
        if GG <= 2*k[0]:
            GG0=GG0+global_hist[k]

        
    print('FG0,HG0,GG0',FG0,HG0,GG0)


    for (x, y), r in kreis_liste2:
        if FG<2*r*pix_um<HG:
            cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 0), 2)
            cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if HG<2*r*pix_um<GG:
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 5)
            cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if GG<2*r*pix_um:
            cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 2)
            cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.putText(image, "{}".format('<300='), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, "{}".format('300<HG<400='), (100, 150),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "{}".format('400<='), (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "{}".format(int(FG0)), (200, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, "{}".format(int(HG0)), (200, 150),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "{}".format(int(GG0)), (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(put+'//'+filename, image)
    return  radius_liste, radius_liste2, Verdachtsfälle
    
def make_local_hist(radius_liste):
        
        binning_list1=[(45,55),(56,70),(71,99),(100,124),(125,149),(150,199),(200,249),(250,299),(300,399),(400,499),(500,600)]
        binning_list=[(k[0]/2,k[1]/2) for k in binning_list1]
        local_hist=dict()
        total_mass=0
        for thresh in binning_list:
                local_hist[thresh]=[0] 
                for radius in radius_liste:
                        
                        if thresh[0]<radius<thresh[1]:
                                if local_hist[thresh]==[0]:
                                        local_hist[thresh]=[3.14*4/3*radius**3]
                                        total_mass=total_mass+3.14*4/3*radius**3
                                else:
                                        local_hist[thresh].append(3.14*4/3*radius**3)
                                        total_mass=total_mass+3.14*4/3*radius**3
                        #print('total_mass,3.14*4/3*radius**3',int(total_mass),int(3.14*4/3*radius**3))
        rel_mass2=0
        for thresh in local_hist:
                rel_mass=0
                for masse in local_hist[thresh]:
                        rel_mass=rel_mass+masse
                rel_mass2=rel_mass2+rel_mass
                
                local_hist[thresh]=rel_mass/total_mass*100
        print('total_mass,rel_mass2',total_mass,rel_mass2)
        return local_hist

foldername='Polydisperse'
take=r'./'+foldername
Analysiert=r'./'+'Analyzed_poly'

pixel_to_um_0=250/78
pixel_to_um_1=250/227
pixel_to_um_2=250/412

pixel_to_um_0_16=250/77*4/1.6
pixel_to_um_1_16=250/227*4/1.6
pixel_to_um_1_16_zoom_out=1000/356
pixel_to_um_1_16_zoom_in=1000/380
pixel_to_um_2_16=250/412*4/1.6



onlyfiles = [f for f in listdir(take) if isfile(join(take, f))]
#p#print.p#print(onlyfiles)
counter=0
global_radius_liste=list()
global_radius_liste2=list()
global_Verdachts_liste=list()
all_local_hists=dict()

for file in onlyfiles:
        counter=counter+1

        
        Bildname=file[0:len(file)-4]
        filename='//'+file
        path=take+filename
        file2=foldername+'_'+Bildname+'_'+'__'+str(counter)+'Poly'+'.JPG'

        radius_liste,radius_liste2,Verdachtsfälle=Find_circles(path,Analysiert,file2,21,51,501,0,20,pixel_to_um_1_16_zoom_in)
        
                
        local_hist=make_local_hist(radius_liste2)
        rel_mass=[local_hist[k] for k in local_hist]
        Durchmesser=[2*(k[0]) for k in local_hist]
        bin_width=[2*(k[1]-k[0]) for k in local_hist]

        fig, ax = plt.subplots()
        rects1 = ax.bar(Durchmesser, rel_mass, width=bin_width , color='yellow',align='edge',edgecolor='black')
        autolabel(rects1)

        Path(Analysiert).mkdir(parents=True, exist_ok=True)
        new_path=Analysiert+'//'+foldername+'_'+Bildname+'_'+str(counter)+'__'+'local_Histogramm_corrected'+'.JPG'

        plt.savefig(new_path)
        plt.clf()
        plt.close()

        for elements in radius_liste2:
                global_radius_liste2.append(elements)

        global_hist2=make_local_hist(global_radius_liste2)
        print('GLOBAL',len(local_hist))

                    
        rel_mass=[global_hist2[k] for k in global_hist2]
        Durchmesser=[2*(k[0]) for k in global_hist2]
        bin_width=[2*(k[1]-k[0]) for k in global_hist2]

        fig, ax = plt.subplots()
        rects1 = ax.bar(Durchmesser, rel_mass, width=bin_width , color='blue',align='edge',edgecolor='black')
        autolabel(rects1)

        new_path=Analysiert+'//'+foldername+'_'+Bildname+'_'+str(counter)+'__'+'GLOBAL_Histogramm_POLY'+'.JPG'

        plt.savefig(new_path)
        plt.clf()
        plt.close()


        all_local_hists[counter]=local_hist

