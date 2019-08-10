# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:02:07 2019

@author: dqs
"""

import numpy as np
import cv2
size = 20
sentence_size=22
def visualization(weight,sentence,num):
    img=np.zeros((55*size,55*size,3), np.uint8)
    for i in range(sentence_size):
        cv2.putText(img,str(i),(10,55+i*size),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(img,str(i),(37+i*size,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        for j in range(sentence_size):
            cv2.rectangle(img,((i+2)*size,(j+2)*size),((i+3)*size,(j+3)*size),(0,0,255*weight[i][j]*10),-1)
    filename="./img/1_img_"+str(num)+".jpg"  
    cv2.imwrite(filename, img)
    #cv2.imshow('image',img)
    #k=cv2.waitKey(0)&0xFF
    #print(k)
    #cv2.destroyAllWindows()
def video():
    fps = 4  
    size_v = (55*size,55*size) 
    videowriter = cv2.VideoWriter("./img/test.avi",cv2.VideoWriter_fourcc('M','J','P','G'),fps,size_v)
     
    for i in range(1,sentence_size):
        filename="./img/a1_img_"+str(i)+".jpg"
        img = cv2.imread(filename)
        videowriter.write(img)
def visual(weight,sentence,num):
    img=np.ones((55*size,55*size,3), np.uint8)
    img=img*255
    str_="."
    for i in range(sentence_size):
        cv2.putText(img,sentence[i],(350,55+i*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+0)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+1)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+2)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    max_len=-1
    sentence_2=[]
    for i in range(sentence_size):
        #print(sentence[i],":",len(sentence[i]))
        if(len(sentence[i])>max_len):
            max_len=len(sentence[i])
    for i in range(sentence_size):
        x=max_len-len(sentence[i])
        str1=" "*x+sentence[i]
        sentence_2.append(str1)
        
    for i in range(sentence_size):
        cv2.putText(img,sentence_2[i],(5,55+i*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+0)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+1)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+2)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    for i in range(sentence_size):
        #cv2.line(img,(max_len*10,55+i*size),(350,55+num*size),(255,255,255*(0.003/weight[num][i])),2)
        cv2.line(img,(max_len*10,55+i*size),(350,55+num*size),(255,255*(0.0025/weight[num][i]),255),2)
    filename="./img_2/1_train_img_"+str(num)+".jpg" 
    cv2.imwrite(filename, img)
    #cv2.imshow('image',img)
    #k=cv2.waitKey(0)&0xFF
    #print(k)
    #cv2.destroyAllWindows()
    
def visual_2(weight,weight1,sentence,num):
    img=np.ones((55*size,55*size,3), np.uint8)
    img=img*255
    str_="."
    for i in range(sentence_size):
        cv2.putText(img,sentence[i],(350,55+i*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+0)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+1)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(370,55+(sentence_size+2)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    max_len=-1
    sentence_2=[]
    for i in range(sentence_size):
        #print(sentence[i],":",len(sentence[i]))
        if(len(sentence[i])>max_len):
            max_len=len(sentence[i])
    for i in range(sentence_size):
        x=max_len-len(sentence[i])
        str1=" "*x+sentence[i]
        sentence_2.append(str1)
        
    for i in range(sentence_size):
        cv2.putText(img,sentence_2[i],(5,55+i*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+0)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+1)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    cv2.putText(img,str_,(105,55+(sentence_size+2)*size),cv2.LINE_AA,0.5,(0,0,0),1)
    for i in range(sentence_size):
        #cv2.line(img,(max_len*10,55+i*size),(350,55+num*size),(255,255,255*(0.003/weight[num][i])),2)
        cv2.line(img,(max_len*10,55+i*size),(350,55+num*size),(255,255*(0.0025/weight[num][i]),255),2)
    cv2.line(img,(max_len*10,55+3*size),(350,55+14*size),(255,255,255*(0.003/weight1[14][3])),2)
    cv2.line(img,(max_len*10,55+10*size),(350,55+14*size),(255,255,255*(0.003/weight1[14][10])),2)
    filename="./img_2/14.jpg" 
    cv2.imwrite(filename, img)
def visualization_2(weight,sentence):
    for i in range(sentence_size):
        visual(weight,sentence,i)
def visualization_3(weight,weight1,sentence):
    for i in range(14,15):
        visual_2(weight,weight1,sentence,i)
if __name__=='__main__':
    """
    img=np.zeros((55*size,55*size,3), np.uint8)   
    #cv2.putText(img,str(1),(0,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1) 
    
    for i in range(53):
        cv2.putText(img,str(i),(10,55+i*size),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(img,str(i),(37+i*size,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        for j in range(53):
            cv2.rectangle(img,((i+2)*size,(j+2)*size),((i+3)*size,(j+3)*size),(0,0,255),-1)
    
    cv2.imshow('image',img)
    k=cv2.waitKey(0)&0xFF
    #print(k)
    cv2.destroyAllWindows()
    """
    img=np.ones((55*size,55*size,3), np.uint8)
    img=img*255
    cv2.imshow('image',img)
    k=cv2.waitKey(0)&0xFF
    #print(k)
    cv2.destroyAllWindows()
    #video()