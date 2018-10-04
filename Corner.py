from PIL import Image, ImageTk
from graphics import GraphWin
from tkFileDialog import askopenfilename,asksaveasfile # Will be used to open the file from the user
import Tkinter
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import sys
from PIL import Image, ImageEnhance
#--------------------------------------------

# Global variables for picture-----------
pic = ''
tkPic = ''
tkPic2 = ''
picToConvert = ''
picWidth = 0
picHeight = 0
canvas1 = ''
def rotateIt(pic1):
    pictureRotated = pic1.rotate(180)
    return pictureRotated

def getRGB(r,g,b):
        red = eval( input ("Enter the value of red: "))
        green = eval(input ("Enter the value of green: "))
        blue = eval(input ("Enter the value of blue: "))
        algorithm =  red-r + green-g + blue-b // 3
        return (algorithm)
# End Gray Algorithms-----------------------------------------------------------------------------

# Draws window, opens picture selected by user, packs the canvas
def drawWindow():
    window = Tkinter.Tk()
    window.title(os.environ.get( "USERNAME" )) # sets the window title to the
    return window

def drawCanvas():
    global window
    global canvas1
    canvas1 = Tkinter.Canvas(window, width = 600, height =300) # Draws a canvas onto the Tkinter window
    canvas1.pack()
    return canvas1

# Global variables for window and canvas
window = drawWindow()
canvas1 = drawCanvas()
# -----------------------------------------------------------------------------------

def openImage():
    global window
    global canvas1
    global pic
    global Oimg
    global Ogray
    global picWidth
    global picHeight
    global tkPic
    global tkPic2
    global picToConvert
    canvas1.delete('all')
    del pic
    del tkPic
    picToConvert = askopenfilename(defaultextension='.jpeg') # Used to open the file selected by the user
    pic = Image.open(picToConvert)
    Oimg=cv2.imread(picToConvert)
    picWidth, picHeight = pic.size # PIL method .size gives both the width and height of a picture
    tkPic = ImageTk.PhotoImage(pic, master = window) # Converts the pic image to a tk PhotoImage
    canvas1.create_image(10,10,anchor='nw', image = tkPic)

def saveImage():
    global pic
    global tkPic2
    pic = Image.open(tkPic2)
    toSave = asksaveasfile(mode='w',defaultextension='.jpeg')
    pic.save(toSave)
def white_hole(img, col,row):
    m=5
    if(row<m):
        return False
    if(col<m):
        return False
    if(col>picWidth-m):
        return False
    if(col>picHeight-m):
        return False
    num=0
    for i in range(row-m,row+m):
        for j in range(col-m,col+m):
            if(img[col][row]==255):
                num+=1
    if(num>4*m*m*0.7):
        return True
    else:
        return False
def eulerToCoordinateTransform(line):
	for rho, theta in line:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
	return [(x1,y1),(x2,y2)]

def point_line(x1,x2,y1,y2,p1,p2):
    A=y2-y1;B=x1-x2;C=-x1*A-y1*B
    d=abs(A*p1+B*p2+C)/np.sqrt(A*A+B*B)
    if(d<30):
        return True
    return False
def getIntersection(line_1, line_2):
	line1 = eulerToCoordinateTransform(line_1)
	line2 = eulerToCoordinateTransform(line_2)
	
	s1 = np.array(line1[0])
	e1 = np.array(line1[1])

	s2 = np.array(line2[0])
	e2 = np.array(line2[1])

	a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
	b1 = s1[1] - (a1 * s1[0])

	a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
	b2 = s2[1] - (a2 * s2[0])

	if abs(a1 - a2) < sys.float_info.epsilon:
		return False

	x = (b2 - b1) / (a1 - a2)
	y = a1 * x + b1
	return (x, y)
def region(x,y):
    if(x+y<picHeight/3):
        return  True
    if(picWidth-x+y<picHeight/3):
        return True
    if(x+picHeight-y<picHeight/3):
        return True
    if(picHeight+picWidth-x-y<picHeight/3):
        return True
    return False
def filter1(edge,x,y):
    m=4
    if(edge[y][x]==0):
        return True
    if(x>m and x<picWidth-m):
        if(y>m and y<picHeight-m):
            for i in range(-m,m+1):
                if(edge[y-m][x+i]==255):
                    return False
                if(edge[y+m][x+i]==255):
                    return False
            for i in range(-m,m+1):
                if(edge[y+i][x-m]==255):
                    return False
                if(edge[y+i][x+m]==255):
                    return False
    return True
def Line(line,bflag=0,fx=0,fy=0,sx=0,sy=0,px=0,py=0,tx=0,ty=0,px1=0,py1=0):
    num=0
    X1=[];Y1=[];X2=[];Y2=[]
    D2=np.sqrt(fx*fx+fy*fy)
    D3=np.sqrt(sx*sx+sy*sy)
    D4=np.sqrt(tx*tx+ty*ty)
    for rho, theta in line:
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        y2 = int(y0 - 1000 * (a))
        dx=x2-x1;dy=y2-y1
        D1=np.sqrt(dx*dx+dy*dy)
        if(bflag==1):
            D=fx*dx+fy*dy
            D=D/D1/D2
            if(abs(D)>0.95):
                continue
        elif(bflag==2):
            D=sx*dx+sy*dy
            DDD=fx*dx+fy*dy
            D=D/D1/D3
            DDD=DDD/D1/D2
            if(abs(D)>0.95 or abs(DDD)>0.99):
                continue
            DD=D_LinePoint(x1,x2,y1,y2,px,py)
            if(DD<60):
                continue
        elif(bflag==3):
            D=sx*dx+sy*dy
            D=D/D1/D3
            if(abs(D)<0.5 or abs(D)>0.99):
                continue
            DDD=fx*dx+fy*dy
            DDD=DDD/D1/D2
            DDDD=tx*dx+ty*dy
            DDDD=DDDD/D1/D4
            if(abs(DDD)>0.992  or abs(DDDD)>0.99):
                continue
            DD=D_LinePoint(x1,x2,y1,y2,px,py)
            DD1=D_LinePoint(x1,x2,y1,y2,px1,py1)
            if(DD<50 or abs(DD1)<50):
                continue
        mm=110
        if(abs(x1-x2)<mm):
            continue
        if(abs(y1-y2)<mm):
            continue
        X1.append(x1);X2.append(x2)
        Y1.append(y1);Y2.append(y2)
        num+=1
    return num,X1,X2,Y1,Y2
def Sect(fx1,fy1,fx2,fy2,sx1,sy1,sx2,sy2):
    a1=fy2-fy1;b1=fx1-fx2;c1=-fx1*(fy2-fy1)+fy1*(fx2-fx1)
    a2=sy2-sy1;b2=sx1-sx2;c2=-sx1*(sy2-sy1)+sy1*(sx2-sx1)
    y=(c2*a1-c1*a2)/(b1*a2-b2*a1)
    x=(-c1-b1*y)/a1
    return x,y
def D_LinePoint(x1,x2,y1,y2,px,py):
    a=y2-y1;b=x1-x2;c=-x1*(y2-y1)+y1*(x2-x1)
    d=np.sqrt(a*a+b*b)
    D=abs(a*px+b*py+c)/d
    return D
def change_pixel():
    global window
    global canvas1
    global tkPic2
    global pic
    # Treats the image as a 2d array, iterates through changing the
    #values of each pixel with the algorithm for gray

    rgbList = pic.load() #Get a 2d array of the pixels
    thresholds1=15;thresholds2=5;thresholds3=100
    Ogray=cv2.cvtColor(Oimg, cv2.COLOR_BGR2GRAY)
    for row in range(picWidth):
        for column in range(picHeight):
            rgb = rgbList[row, column]
            if(picHeight<200):
                if(column<picHeight/5):
                    Ogray[column][row] = 255
                    continue
            else:
                if(column<picHeight/12*5-10):
                    Ogray[column][row] = 255
                    continue
                #rgbList[row, column] = (Ogray[column][row], Ogray[column][row], Ogray[column][row])  # Gives each pixel a new RGB value
                
            if(region(row,column)):
                Ogray[column][row] = 255
            # print rgb
            rgb1 = rgbList[row, column-3]
            r1, g1, b1 = rgb1
            r, g, b = rgb
            if (r<thresholds3 or g<thresholds3 or b<thresholds3):
                Ogray[column][row] = 255
                continue
            if(abs(r-g)>thresholds1 or abs(r-b)>thresholds1 or abs(g-b)>thresholds1):
                Ogray[column][row] = 255
                #rgbList[row, column] = (Ogray[column][row], Ogray[column][row], Ogray[column][row])  # Gives each pixel a new RGB value
                continue
            if(abs(r-r1)<thresholds2 or abs(g1-g)<thresholds2 or abs(b-b1)<thresholds2):
                Ogray[column][row] = 255
    edges = cv2.Canny(Ogray, 30, 40, apertureSize=3)
    for row in range(picWidth):
        for column in range(picHeight):
            if(filter1(edges,row,column)):
                edges[column][row]=0

        # Converting to a Tkinter PhotoImage
    #first line
    flag=1
    X1=[];X2=[];Y1=[];Y2=[]
    thres=100
    while(flag==1):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thres)
        L,X1,X2,Y1,Y2=Line(lines[0])
        if( L==1):
            flag=0
        else:
            if(L<10):
                thres+=1
            else:
                thres+=10
    cv2.line(Oimg, (X1[0], Y1[0]), (X2[0], Y2[0]), (0, 0, 255), 2)
    #cv2.imshow("Line Detection", Oimg)
    #cv2.waitKey(0)
    fx1=X1[0];fx2=X2[0];fy1=Y1[0];fy2=Y2[0]
    dfx=fx2-fx1;dfy=fy2-fy1
    #second line
    flag=1
    thres=60
    while(flag==1):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thres)
        L,S1,S2,T1,T2=Line(lines[0],1,dfx,dfy)
        if( L==1):
            flag=0
        else:
            if(L<20):
                thres+=1
            else:
                thres+=10
    

    cv2.line(Oimg, (S1[0], T1[0]), (S2[0], T2[0]), (0, 0, 255), 2)
    #cv2.imshow("Line Detection", Oimg)
    #cv2.waitKey(0)
    sx1=S1[0];sx2=S2[0];sy1=T1[0];sy2=T2[0]
    dsx=sx2-sx1;dsy=sy2-sy1
    #third line
    px,py=Sect(fx1,fy1,fx2,fy2,sx1,sy1,sx2,sy2)
    flag=1
    thres=50
    while(flag==1):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thres)
        L,U1,U2,V1,V2=Line(lines[0],2,dfx,dfy,dsx,dsy,px,py)
        if( L==1):
            flag=0
        else:
            if(L<30):
                thres+=1
            else:
                thres+=5
    

    cv2.line(Oimg, (U1[0], V1[0]), (U2[0], V2[0]), (0, 0, 255), 2)
    tx1=U1[0];tx2=U2[0];ty1=V1[0];ty2=V2[0]
    dtx=tx2-tx1;dty=ty2-ty1
    px1, py1 = Sect(tx1, ty1, tx2, ty2, sx1, sy1, sx2, sy2)
    bflag=0
    if(px1>picWidth-1 or py1>picHeight):
        px1, py1 = Sect(tx1, ty1, tx2, ty2, fx1, fy1, fx2, fy2)
        bflag=1
    #cv2.imshow("Line Detection", Oimg)
    #cv2.waitKey(0)

    #forth line
    flag=1
    thres=60
    while(flag==1):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thres)
        L,P1,P2,Q1,Q2=Line(lines[0],3,dfx,dfy,dsx,dsy,px,py,dtx,dty,px1,py1)
        if( L==1):
            flag=0
        else:
            if(L<40):
                thres+=1
            else:
                thres+=5
    cv2.line(Oimg, (P1[0], Q1[0]), (P2[0], Q2[0]), (0, 0, 255), 2)
    ex1 =P1[0];ex2 = P2[0];ey1 = Q1[0];ey2 = Q2[0]
    if(bflag==0):
        px2, py2 = Sect(tx1, ty1, tx2, ty2, ex1, ey1, ex2, ey2)
        px3, py3 = Sect(fx1, fy1, fx2, fy2, ex1, ey1, ex2, ey2)
    else:
        px2, py2 = Sect(tx1, ty1, tx2, ty2, ex1, ey1, ex2, ey2)
        px3, py3 = Sect(sx1, sy1, sx2, sy2, ex1, ey1, ex2, ey2)
    # Show the result
    cv2.imshow("Line Detection", Oimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #rgbList[px, py]=(255,0,0);rgbList[px1, py1]=(255,0,0);rgbList[px2, py2]=(255,0,0);rgbList[px3, py3]=(255,0,0)
    print("First point:",px,py)
    print("Second point:", px1, py1)
    print("Third point:", px2, py2)
    print("Forth point:", px3,py3)

    #plt.imshow(Oimg,'gray')
    #plt.show()
    #plt.imshow(edges,'gray')
    #plt.show()
    #del tkPic2
    #tkPic2 = ImageTk.PhotoImage(pic, master = window)
    #canvas1.create_image(510,260, anchor='ne',image = tkPic2)

# Function to create a button, takes the button text and the function to be called on click
def tkButtonCreate(text, command):
    Tkinter.Button(window, text = text, command = command).pack()

def main():
    tkButtonCreate('Open Image',openImage)
    tkButtonCreate('Corner detection', change_pixel)
    #tkButtonCreate('Save',saveImage)
    window.mainloop()
    #convertButton = Tkinter.Button(window,text = 'Convert', command = change_pixel).pack()
main()