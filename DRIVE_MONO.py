
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



def autolabel(rects):                                                   #source: [.........]
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % height.round(1),
                ha='center', va='bottom')
                
def make_global_hist(radius_list):                                      #source[self]
        print('create a dictionary: keys=corn size range, values= radia')
        binning_list1=[(45,55),(56,70),(71,99),(100,124),(125,149),(150,199),(200,249),(250,299),(300,399),(400,419),(420,449),(450,499),(500,600)]#[(int(max_Rad*i/n),int(max_Rad*(i+1)/n)) for i in range(n)]
        binning_list=[(k[0]/2,k[1]/2) for k in binning_list1]
        local_hist=dict()
        for thresh in binning_list:
                local_hist[thresh]=[0] 
                for radius in radius_list:
                        if thresh[0]<radius<thresh[1]:
                                if local_hist[thresh]==[0]:
                                        local_hist[thresh]=[radius]                                       
                                else:
                                        local_hist[thresh].append(radius)
        return local_hist

                
def Find_circles(source_directory,put,filename,SF1,SF2,KS,C,minDist,pix_um,overlap):   
    #source_directory, put, filename= directories for source and store directory
    #SF1,SF2= meanshift kernel size (here: 21,51)
    
    image = cv2.imread(source_directory)
    print('OPENED',source_directory)
    
    print('################# level out global gradient and transform to binary #######')
    shifted = cv2.pyrMeanShiftFiltering(image, SF1,SF2)#21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                  cv2.THRESH_BINARY,KS,C)#501,10)
    print('####create maxima by distance transform in the center of the beads and watershed ######')
    D = ndimage.distance_transform_edt(thresh)                                      #thin regions become darker
    localMax = peak_local_max(D, indices=False, min_distance=minDist,labels=thresh) 
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]         
    labels = watershed(-D, markers, mask=thresh)                                    #separate particles
    print('################ find OBJECTS ####################')
    all_circles_list=list()
    all_radia_list=list()
    all_circles_DICT=dict()
    for label in np.unique(labels):
            if label == 0:
                    continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,                 #create contourgroups
                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)                                      #center of countourgroup
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            if r>minDist:
                all_circles_list.append((label,(x, y), r))
                all_circles_DICT[label]=[(x, y), r]
                all_radia_list.append(r*pix_um)

    print('###########################################################################')
    print('############################ Sort out #####################################')
    print('###########################################################################')
    
    
    print('identify strongly overlapping circles; too much overlapp is a sign for failed center or countour detection')
    overlapping_radia_list=list()
    overlapping_circles_list=list()
    overlapping_circles_DICT=dict()
    for label1,(x1, y1), r1 in all_circles_list:
        for label,(x, y), r in all_circles_list:
            dist=((x-x1)**2+(y-y1)**2)**0.5
            if  [(label,(x, y), r),(label1,(x1,y1), r1)] not in overlapping_circles_list and [(label1,(x1, y1), r1),(label,(x,y), r)] not in overlapping_circles_list and label!=label1:
                if dist<overlap*(r+r1):
                    overlapping_radia_list.append(r)
                    overlapping_radia_list.append(r1)
                    overlapping_circles_list.append([(label,(x, y), r),(label1,(x1,y1), r1)])
                    overlapping_circles_DICT[label]=[(x, y), r]
                    overlapping_circles_DICT[label1]=[(x1, y1), r1]
                   
    print('### collect circles that do not overlap more than overlap in %')
    cleaned_radius_list=list()
    cleaned_circle_list=list()
    cleaned_circles_DICT=dict()
    for label in all_circles_DICT:
        if label not in overlapping_circles_DICT:
            [(x, y), r]=all_circles_DICT[label]
            cleaned_radius_list.append(r)
            cleaned_circle_list.append((label,(x, y), r))
            cleaned_circles_DICT[label]=[(x, y), r]
    
    
    print('### correction for weakly overlapping circles/// ideal in case of monodisperse sample #### shrink circles proportional to their radius ### until they dont overlap anymore')

    for label in cleaned_circles_DICT:
        for label1 in cleaned_circles_DICT:
            [(x, y), r]=cleaned_circles_DICT[label]
            [(x1, y1), r1]=cleaned_circles_DICT[label1]
            if label!=label1:
                dist=((x-x1)**2+(y-y1)**2)**0.5
                if r+r1>dist>overlap*(r+r1):
                    delta=abs(r+r1-dist)
                    r=r-r/(r1+r)*delta
                    r1=r1-r1/(r1+r)*delta
                    cleaned_circles_DICT[label]=[(x, y), r]
                    cleaned_circles_DICT[label1]=[(x1, y1), r1]

    print('### Convert from DICT to list')
    corrected_circles_list=list()
    corrected_radia_list=list()
    for label in cleaned_circles_DICT:
        [(x, y), r]=cleaned_circles_DICT[label]
        corrected_circles_list.append((label,(x, y), r))
        corrected_radia_list.append(r*pix_um)
        
    print('### show circles on image')
    for label,(x, y), r in corrected_circles_list:
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    for (label,(x, y), r),(label1,(x1,y1), r1) in overlapping_circles_list:
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 10)
        cv2.putText(image, "{}".format(int(r*pix_um)), (int(x) - 10, int(y)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(image, (int(x1), int(y1)), int(r1), (0, 0, 255), 10)
        cv2.putText(image, "{}".format(int(2*r*pix_um)), (int(x) - 10, int(y)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(image, (int(x), int(y)), (int(x1),int(y1)), (0, 0, 255), 3)
        
        
    cv2.imwrite(put+'//'+filename, image)
    return  corrected_radia_list
    

################################################################################################################################
################################# Execute ######################################################################################
################################################################################################################################

#Paths
source_folder='Monodisperse_Input_images'
source_directory=r'./'+source_folder
destination=r'./'+'Analyzed_Mono'
#Magnification factors: pixel to µm
pixel_to_um_0=250/78
pixel_to_um_1=250/227
pixel_to_um_2=250/412
pixel_to_um_0_16=250/77*4/1.6
pixel_to_um_1_16=250/227*4/1.6
pixel_to_um_1_16_zoom_out=1000/356
pixel_to_um_1_16_zoom_in=1000/380
pixel_to_um_2_16=250/412*4/1.6

onlyfiles = [f for f in listdir(source_directory) if isfile(join(source_directory, f))]

counter=0
global_radius_list=list()


for file in onlyfiles:
        counter=counter+1
        pic_name=file[0:len(file)-4]
        filename='//'+file
        take=source_directory+filename
        new_pic_name=source_folder+'_'+pic_name+'_'+'__'+str(counter)+'.JPG'
        
        print('##############Find_circles##########################')
        radius_list=Find_circles(take,destination,new_pic_name,21,51,501,0,20,pixel_to_um_1_16_zoom_in,0.6)

        print('#collect radia for histogramm')
        for elements in radius_list:
                global_radius_list.append(elements)
                
        print('#create mass distribution from radia ')
        global_hist=make_global_hist(global_radius_list)
        mass_global=list()
        mass_global_dict=dict()
        for k in global_hist:
            if global_hist[k]!=[0]:
                mass_global.append(3.14*4/3*(k[1]-abs(k[0]-k[1])/2)**3*len(global_hist[k]))
                mass_global_dict[k]=3.14*4/3*(k[1]-abs(k[0]-k[1])/2)**3*len(global_hist[k])
            else:
                mass_global.append(0)
                mass_global_dict[k]=0
                
        #normalize to toal mass
        total_mass=0
        for masse in mass_global:
                total_mass=total_mass+masse
                
        rel_mass_global_dict=dict()
        for k in mass_global_dict:
                rel_mass_global_dict[k]=mass_global_dict[k]/total_mass
        
        # plot histogramm
        rel_mass=[rel_mass_global_dict[k]*100 for k in rel_mass_global_dict]
        Durchmesser=[2*(k[0]) for k in rel_mass_global_dict]
        bin_width=[2*(k[1]-k[0]) for k in rel_mass_global_dict]
        fig, ax = plt.subplots()
        rects1 = ax.bar(Durchmesser, rel_mass, width=bin_width , color='green',align='edge',edgecolor='black')
        autolabel(rects1)

        # save the histogramm
        Path(destination).mkdir(parents=True, exist_ok=True)
        new_path=destination+'//'+source_folder+'_'+pic_name+'_'+str(counter)+'__'+'GLOBAL_Histogramm'+'.JPG'
        plt.savefig(new_path)
        plt.clf()
        plt.close()

        


###Save in an excellsheet
Excell_liste=[[2*k[0],rel_mass_global_dict[k]*100] for k in rel_mass_global_dict]
with open(foldername+'_'+pic_name+'__'+str(counter)+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(Excell_liste)
