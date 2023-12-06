#square polygon face construction
import os
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread 
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndimage 
from skimage.io import imread

inputPath="lung/normal/train"
outputPath="lungpoint/normal/train"
cnt=0
for filename in os.listdir(inputPath):
  print(cnt)
  cnt+=1
  imageFile=inputPath+"/"+filename
  hsv_img = imread(imageFile)
  #hsv_img = rgb2hsv(im)
  dim=224
  l=[]
  for i in range(dim):
    temp=[]
    for j in range(dim):
      temp.append(0.2126*int(hsv_img[i][j][0])+0.7152*int(hsv_img[i][j][1])+0.0722*int(hsv_img[i][j][2]))
      #temp.append(0.299*int(hsv_img[i][j][0])+0.587*int(hsv_img[i][j][1])+0.114*int(hsv_img[i][j][2]))
    l.append(temp)


  off=[]

  points=dim*dim+4*dim
  planes=(dim-1)*(dim-1) + 4*(dim-1) + 1

  for i in range(dim):
    for j in range(dim):
      off.append([i,j,l[i][j]])

  mini=0



  for i in range(dim):
    off.append([0,i,mini])
  for i in range(dim):
    off.append([i,0,mini])
  for i in range(dim):
    off.append([i,dim-1,mini])
  for i in range(dim):
    off.append([dim-1,i,mini])


  for i in range(dim-1):
    for j in range(dim-1):
      off.append([4 , (i*dim+j) , (i*dim+j+1) , ((i+1)*dim+j+1) , ((i+1)*dim+j)])


  #back side
  off.append([4,dim*dim,dim*dim+dim-1,dim*dim+3*dim-1,dim*dim+2*dim-1])

  #left side
  for i in range(dim-1):
    off.append([4,i*dim, (i+1)*dim, dim*dim+dim+i+1 ,dim*dim+dim+i])

  #right side
  for i in range(dim-1):
    off.append([4, (i+1)*dim-1, (i+2)*dim-1, dim*dim+2*dim+i+1, dim*dim+2*dim+i ])

  #up side
  for i in range(dim-1):
    off.append([4, i, i+1, dim*dim+i+1, dim*dim+i])

  #down side
  for i in range(dim-1):
    off.append([4, dim*(dim-1)+i , dim*(dim-1)+i+1 ,dim*dim+3*dim+i+1, dim*dim+3*dim+i])

  output=outputPath+"/"+filename[:-3:]+"off"
  f= open(output,"w+")
  f.write("OFF")
  f.write("\n")
  f.write(str(points) + " " + str(planes) + " " + str(0))
  f.write("\n")

  for i in range(points):
    f.write(str(off[i][0])+ " " + str(off[i][1]) + " " + str(off[i][2]))
    f.write("\n")

  for i in range(points,points+planes):
    #print(off[i][0], off[i][1], off[i][2], off[i][3])
    f.write(str(off[i][0])+ " " + str(off[i][1]) + " " + str(off[i][2]) + " " + str(off[i][3])  + " " +str(off[i][4]))
    f.write("\n")

  f.close()

        
        

