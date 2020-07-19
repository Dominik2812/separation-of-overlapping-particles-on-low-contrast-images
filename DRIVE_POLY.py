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


def autolabel(rects):#source: [.........]
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
                
def make_local_hist(radius_list):
        print('create a dictionary: keys=corn size range, values= radia')
        binning_list1=[(45,55),(56,70),(71,99),(100,124),(125,149),(150,199),(200,249),(250,299),(300,399),(400,499),(500,600)]
        binning_list=[(k[0]/2,k[1]/2) for k in binning_list1]
        local_hist=dict()
        total_mass=0
        for thresh in binning_list:
                local_hist[thresh]=[0] 
                for radius in radius_list:
                        
                        if thresh[0]<radius<thresh[1]:
                                if local_hist[thresh]==[0]:
                                        local_hist[thresh]=[3.14*4/3*radius**3]
                                        total_mass=total_mass+3.14*4/3*radius**3
                                else:
                                        local_hist[thresh].append(3.14*4/3*radius**3)
                                        total_mass=total_mass+3.14*4/3*radius**3
        rel_mass2=0
        for thresh in local_hist:
                rel_mass=0
                for masse in local_hist[thresh]:
                        rel_mass=rel_mass+masse
                rel_mass2=rel_mass2+rel_mass
                
                local_hist[thresh]=rel_mass/total_mass*100
        print('total_mass,rel_mass2',total_mass,rel_mass2)
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

    cleaned_radia_list=[r*pix_um for r in all_radia_list if r not in overlapping_radia_list]
    cleaned_circle_list=[(label,(x, y), r) for (label,(x, y), r) in all_circles_list if r not in overlapping_radia_list]
    
    
    print('### !!!!!!!!! preparation for visualisation of 3 classes of particles, FG=below ,//// HG within ///GG= above a desired size')
    global_hist=make_local_hist(cleaned_radia_list)

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

    print('### !!!!!!!!! Visualisation of 3 classes of particles, FG, HG,GG')
    for label,(x, y), r in cleaned_circle_list:
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
    return  all_radia_list, cleaned_radia_list, overlapping_radia_list



################################################################################################################################
################################# Execute ######################################################################################
################################################################################################################################


#Paths
source_folder='Polydisperse_Input_images'
source_directory=r'./'+source_folder
destination=r'./'+'Analyzed_Poly'
#Magnification factors: pixel to µm
pixel_to_um_0=250/78
pixel_to_um_1=250/227
pixel_to_um_2=250/412
pixel_to_um_0_16=250/77*4/1.6
pixel_to_um_1_16=250/227*4/1.6
pixel_to_um_1_16_zoom_out=1000/356
pixel_to_um_1_16_zoom_in=1000/380
pixel_to_um_2_16=250/412*4/1.6

pics = [f for f in listdir(source_directory) if isfile(join(source_directory, f))]


counter=0
global_radius_liste=list()
global_radia_list=list()
global_Verdachts_liste=list()
all_local_hists=dict()

for pic in pics:
        counter=counter+1
        pic_name=pic[0:len(pic)-4]
        filename='//'+pic
        take=source_directory+filename
        new_pic_name=source_folder+'_'+pic_name+'_'+'__'+str(counter)+'.JPG'
        
        print('##############Find_circles##########################')
        all_radia_list, cleaned_radia_list, overlapping_radia_list=Find_circles(take,destination,new_pic_name,21,51,501,0,20,pixel_to_um_1_16_zoom_in,overlap=0.6)
        
        print('# LOCAL histogramm')
        local_hist=make_local_hist(cleaned_radia_list)
        rel_mass=[local_hist[k] for k in local_hist]
        diameter=[2*(k[0]) for k in local_hist]
        bin_width=[2*(k[1]-k[0]) for k in local_hist]
        fig, ax = plt.subplots()
        rects1 = ax.bar(diameter, rel_mass, width=bin_width , color='yellow',align='edge',edgecolor='black')
        autolabel(rects1)
        
        Path(destination).mkdir(parents=True, exist_ok=True)
        new_path=destination+'//'+source_folder+'_'+pic_name+'_'+str(counter)+'__'+'local_Histogramm_corrected'+'.JPG'

        plt.savefig(new_path)
        plt.clf()
        plt.close()
        
        
        
        print('# GLOBAL histogramm')
        for elements in cleaned_radia_list:
                global_radia_list.append(elements)

        global_hist=make_local_hist(global_radia_list)
        print('GLOBAL',len(local_hist))

                    
        rel_mass=[global_hist[k] for k in global_hist]
        diameter=[2*(k[0]) for k in global_hist]
        bin_width=[2*(k[1]-k[0]) for k in global_hist]

        fig, ax = plt.subplots()
        rects1 = ax.bar(diameter, rel_mass, width=bin_width , color='blue',align='edge',edgecolor='black')
        autolabel(rects1)

        new_path=destination+'//'+source_folder+'_'+pic_name+'_'+str(counter)+'__'+'GLOBAL_Histogramm_POLY'+'.JPG'

        plt.savefig(new_path)
        plt.clf()
        plt.close()


        all_local_hists[counter]=local_hist

