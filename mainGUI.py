# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:14:40 2020

@author: z028041
"""
from tkinter import * 
from PIL import ImageTk,Image  
import random
import tkinter.simpledialog
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage import filters
import os
import cv2
import urllib.request
from tkinter import *
from PIL import ImageFilter,Image
from tkinter import filedialog, messagebox
import os
import time
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


def display_image(input_image):
	print("display_image called")
	canvas.delete("pyramid")
	path = input_image
	global img
	img = ImageTk.PhotoImage(Image.open(path))
	canvas.itemconfig(img_on_can,image = img)
	return 
def show_entry_fields():
    print ('Hello Ur Details Enterd ...!!! ')
    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
    print("User ID %s" % e3.get())
    print("Rating: %s" % (e4.get()))
    print("Age: %s" % (e5.get()))
    return e1.get(), e2.get(),e3.get(),e4.get(),e5.get()

def concat(image):
    Name1,Name2,UserId,RatExpInp1,age=show_entry_fields()  
    age=int(age)
    print (Name1,Name2,RatExpInp1)
    #print (Name1,Name2,RatExpInp1)
    RatExpInp=RatExpInp1
    ## Reading of the Dataet
    data=pd.read_csv('books-1.csv')
    #print(data.head(5))
    
    ## Year Wise Separation
    original_publication_yeardata=data['original_publication_year']
    isbndata=data['isbn']
    yearseparated=[]
    yearseparatedID=[]
    original_publication_yeardataNumpy=original_publication_yeardata.to_numpy()
    for i in range(len(original_publication_yeardata)):
      if ((original_publication_yeardataNumpy[i]>1900)and(original_publication_yeardataNumpy[i]<2020)):
        yearseparated.append(isbndata[i])
        yearseparatedID.append(i)
    
    # retrieving rows by loc method
    yearseparatedIDdata = data.iloc[yearseparatedID]
    print ('year wise Separation Of Data Completed !!!')
    #print (yearseparatedIDdata)
    
    ##Book ID checking
    isbndata=data['book_id']
    isbnseparated=[]
    isbnseparatedID=[]
    
    isbndatanumpy=isbndata.to_numpy()
    for i in range(len(isbndata)):
      if math.isnan(isbndatanumpy[i]):
        print('NAN Detected')
      else :
        isbnseparated.append(isbndatanumpy[i])
        isbnseparatedID.append(i)
    
    # retrieving rows by loc method 
    print ('ISBN NaN Data Removal Info')
    data = data.iloc[isbnseparatedID]
    #    print (len(data))
    
    ## K-Means Based Review Count wise Ranking
    import matplotlib.pyplot as plt3
    ratings_count_data=data['ratings_count']
    
    Datacalorie=np.array(ratings_count_data)
    X=np.reshape(Datacalorie,(-1,1))
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    #print ('## Prediction Result ##')
    #print (kmeans.predict([[Datacalorie[0]]]))
    XValu=np.arange(0,len(kmeans.labels_))
    #print (len(kmeans.labels_))
    
    fig,axs=plt3.subplots(1,1,figsize=(15,5))
    plt3.bar(XValu,kmeans.labels_)
    plt3.show()
    ClusteredData=kmeans.labels_
    posapp=[]
    valuapp=[]
    
    
    ## Reading of the datadfet
    datadf=pd.read_csv('age_ratings.csv')
    print(datadf.head(5))
    
    
    ## age Wise Separation
    
    original_publication_agedatadf=datadf['age']
    print (original_publication_agedatadf)
    isbnagedatadf=datadf['book_id']
    ageseparated=[]
    ageseparatedID=[]
    original_publication_agedatadfNumpy=original_publication_agedatadf.to_numpy()
    for i in range(len(original_publication_agedatadf)):
      if ((original_publication_agedatadfNumpy[i]>0)and(original_publication_agedatadfNumpy[i]<100)):
        ageseparated.append(isbnagedatadf[i])
        ageseparatedID.append(i)
    
    # retrieving rows by loc method
    ageseparatedIDdatadf = datadf.iloc[ageseparatedID]
    print ('age wise Separation Of datadf Completed !!!')
    #print (ageseparatedIDdatadf)
    
    
    
    ## AGE WISE RATING SEPARATION
    datadf=pd.read_csv('age_ratings.csv')
    databk=pd.read_csv('books-1.csv')
    my_data=datadf
    bookval=datadf['book_id']
    groups = my_data.groupby('book_id')
    print(bookval[1])
    
    Grp_avail1=groups.get_group(bookval[0]);
    Grp_avail2=groups.get_group(bookval[1]);
    lengrp=len(groups)
    print (Grp_avail2)
    grpdt=[]
    bookvalid=[]
    Gmean=np.zeros((lengrp,),dtype=np.uint8)
    for j in range(lengrp):
        Grp_avail=groups.get_group(bookval[j])
        Grp_avail=Grp_avail.T
        G2=math.ceil(math.ceil(np.mean(Grp_avail.iloc[1]))/10)
        grpdt.append(G2)
        bookvalid.append(bookval[j])
    #    print (bookval[j],G2)
    
    print ('Overall Age wise Rated Books count ...',len(grpdt))
    
    import matplotlib.pyplot as plt6
    plt6.bar(bookvalid,grpdt,facecolor='g')
    plt6.grid(True)
    plt6.title('Age Wise Book aggregations')
    plt6.show()
    # datadf type conversions
    # Fixing random state for reproducibility
    
    x = isbnagedatadf
    y=original_publication_agedatadf
    
    import matplotlib.pyplot as plt7
    plt7.scatter(x,y, facecolor='r')
    plt7.xlabel('book')
    plt7.ylabel('age')
    plt7.title('Histogram of books')
    
    #conditions  
    print ('######################')
    agecl=math.ceil(age/10)
    print (agecl)   
    
    
    # #plot datadf
    # fig, ax = plt.subplots(figsize=(15,7))
    # datadf.groupby(['age','book_id']).count()['book_id'].plot(ax=ax)
    
    for i in range(len(ClusteredData)):
        if ClusteredData[i] >2:
            posapp.append(isbndata[i])
            valuapp.append(i)
    data = data.iloc[posapp]
    #    print (data)
    
    ## K-Means Based Review Rating wise Ranking
    import matplotlib.pyplot as plt5
    ratings_count_data=data['average_rating']
    Datacalorie=np.array(ratings_count_data)
    X=np.reshape(Datacalorie,(-1,1))
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    print ('## Prediction Result ##')
    #    print (kmeans.predict([[Datacalorie[0]]]))
    XValu=np.arange(0,len(kmeans.labels_))
    finX=agecl
    #    print (len(kmeans.labels_))
    
    fig,axs=plt5.subplots(1,1,figsize=(15,5))
    plt5.bar(XValu,kmeans.labels_)
    plt5.title("Review Rating wise Separated")
    print(("Review Rating wise Separation completed !!!"))
    
    import matplotlib.pyplot as plt6
    ClusteredRateData=kmeans.labels_
    posapp=[]
    valuapp=[]
    brklbl=[]
    
    for i in range(len(ClusteredRateData)):
        if ClusteredRateData[i] >1:
            posapp.append(ClusteredRateData[i])
            valuapp.append(i)
            brklbl.append(ClusteredRateData[i])
    plt6.title("Avereage Ratings sepeartion completed")
    kmeansRevCntIDdata = data.iloc[valuapp]
    print (len(kmeansRevCntIDdata))
    
    XValu=np.arange(0,len(posapp))
    fig,axs=plt.subplots(1,1,figsize=(15,5))
    plt6.bar(XValu,posapp)
    plt6.title("Count K- Means Group Separated")
    data = data.iloc[valuapp]
    
        
    RatExpInp=int(RatExpInp)
    print (RatExpInp)
    if (RatExpInp==9):
        revclsInp=1
        print ('OKay')
    elif ( RatExpInp==8):
        revclsInp=1
    elif ( RatExpInp==7):
        revclsInp=2
    elif ( RatExpInp==6):
        revclsInp=2
    elif ( RatExpInp==5):
        RatExpInp=3
    elif ( RatExpInp==4):
        revclsInp=3
    elif ( RatExpInp==3):
        revclsInp=4
    elif ( RatExpInp==2):
        revclsInp=4
    valfin=revclsInp
    print ('Classification Data Preprocessing initiated !!!')
    revcls=[1,2,3,4,5]
    datafinext=data.iloc[:,8:10]
    
    
    print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    bookidDet=data.iloc[:,1]
    bookidDet=list(bookidDet)
    agedatfin=np.zeros(len(bookidDet),dtype=np.uint8)
    
    
    for ik in range(len(bookidDet)):
        for ij in range(len(bookvalid)):
            x=bookidDet[ik]
            y=bookvalid[ij]
            if x==y:
                agedatfin[ik]=grpdt[ij]
                break
            else:
                agedatfin[ik]=0
    
    # code to replace all negative value with 0 
    if agecl <3:
        agedatfin[agedatfin==0] = 2
    else:
        agedatfin[agedatfin==0] = 6
    datafinextNumpy=datafinext.to_numpy()
    #    print (datafinextNumpy)
    
    datafin=np.zeros((len(datafinext),3),dtype=np.float32)
    #weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    t=0
    yt=[]
    for jj in range(len(agedatfin)):
        valloc=list(datafinextNumpy[jj])
        valloc.append(agedatfin[jj])
        datafin[jj]=np.array(valloc)
        RatExp=brklbl[jj]
        if (RatExp==9):
            revcls=4
        elif ( RatExp==8):
            revcls=4
        elif ( RatExp==7):
            revcls=3
        elif ( RatExp==6):
            revcls=3
        elif ( RatExp==5):
            revcls=2
        elif ( RatExp==4):
            revcls=2
        elif ( RatExp==3):
            revcls=1
        elif ( RatExp==2):
            revcls=1
        yt.append(revcls)
        # X_test=np.zeros((len(data),6),dtype=np.float32)

    datafinTest=np.zeros((len(datafinext),3),dtype=np.float32)
    #weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    t=0
    for jj in range(len(agedatfin)):
        valloc=list(datafinextNumpy[jj])
        valloc.append(agedatfin[jj])
        datafinTest[jj]=np.array(valloc)*finX
        RatExp=brklbl[jj]
        if (RatExp==9):
            revcls=4
        elif ( RatExp==8):
            revcls=4
        elif ( RatExp==7):
            revcls=3
        elif ( RatExp==6):
            revcls=3
        elif ( RatExp==5):
            revcls=2
        elif ( RatExp==4):
            revcls=2
        elif ( RatExp==3):
            revcls=1
        elif ( RatExp==2):
            revcls=1  

    X_test=datafinTest   
    from sklearn.model_selection import train_test_split
    X=datafin# Features
    y=yt # Labels
    print ('$$$$$$$$$$$$$$$$$$$$$$$$$')
    print ("Number Of Labels :: ")
    print (len(yt))
    print ("Number Of Train Data :: ")
    print (len(X))
    print ('Classifier Loaded !!!')
    # Split dataset into training set and test set
    X_train= X# Features
    y_train= yt # Labels
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print ('Printing Results !!!')
    print ('SUGGESTED BOOKS ITEMS {Showing Top 10}:')
    TitlesSuggest=data['original_title']
    TitlesSuggestNumpy=TitlesSuggest.to_numpy()
    image_url=data['image_url']
    image_urlNumpy=image_url.to_numpy()
    
    pr10=0
    imgsLinks=[]
    for ii in range(len(y_pred)):
        if y_pred[ii]==valfin:     #weightloss
            if pr10<10:
                print('#################')
                print (TitlesSuggestNumpy[ii])
                print (image_urlNumpy[ii])
                imgsLinks.append(image_urlNumpy[ii])
            pr10+=1
    
    tempimg=cv2.imread('tempimg.JPG')
    tempimg=cv2.resize(tempimg,(20,150))
    path='./datasetsbooks/'
    cnt=0
    image=imgsLinks
    for i in image:
        if cnt==0:
            req = urllib.request.urlopen(i)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            im_h = cv2.imdecode(arr, -1) # 'Load it as it is'
            im_h=cv2.resize(im_h,(80,150))
        else:
            req = urllib.request.urlopen(i)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            temp = cv2.imdecode(arr, -1) # 'Load it as it is'            
            temp=cv2.resize(temp,(80,150))
            im_h = cv2.hconcat([im_h,tempimg,temp])
        cnt+=1
    img_path='1.jpg'
    cv2.imwrite('1.jpg', im_h)
    display_image(img_path)
    


root = Tk()  
# #  image_path = ["1(1).jpg","1(2).jpg","1(3).jpg","1(4).jpg","1(5).jpg","1(6).jpg","1(7).jpg","1(8).jpg","1(9).jpg","1(10).jpg"] # <--- Direct Images name here
image_path=''
#image_path=[]
#for lp in range(1,11):
#    image_path.append(str(lp)+'.jpg') # <--- ID based Images name generation
#print (image_path[0])
canvas = Canvas(root, width = 1600, height = 400)  
canvas2 = Canvas(root, width = 300, height = 200)  
canvas2.pack()  

canvas.pack()  
image_pathfin='orig.JPG'
test_image = ImageTk.PhotoImage(Image.open(image_pathfin))  
img_on_can = canvas.create_image(40, 60, anchor=NW, image=test_image) 
canvas.image = test_image
im = cv2.imread(image_pathfin)

#master = tk.Tk()
Label(canvas2,text="First Name").grid(row=0,column=0)
Label(canvas2,text="Last Name").grid(row=1,column=0)
Label(canvas2,text="User Id").grid(row=2,column=0)
Label(canvas2,text="Rating").grid(row=3,column=0)
Label(canvas2,text="Age").grid(row=4,column=0)


e1 = Entry(canvas2)
e2 = Entry(canvas2)
e3 = Entry(canvas2)
e4 = Entry(canvas2)
e5 = Entry(canvas2)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)


button = Button(root,text="Search",command = lambda: concat(image_path))
button = canvas.create_window(700,20, anchor=NW,window=button)
root.mainloop()  