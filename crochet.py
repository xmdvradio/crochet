import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter.font import Font
from tkinter import filedialog
import time
from PIL import ImageTk, Image
import csv
import threading
from random import randint

class frame():
    def __init__(self, f):
        #variables
        self.w = self.h = 1

        #processed images
        self.f = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        self.p = None
        self.colours = None
        self.counts = None
        self.i = None
        self.indexes = None
        self.pallet = None
        self.npal = None
        
    def generatecolours(self):
        #variable setup
        f = self.f

        start = time.time()
        #if colours are weird check that its rotating the histogram correctly
        f = np.reshape(f, (-1,3)) #flatten the image to a big line of pixles
        f = np.rot90(f, -1) #make what 3 big long lists -> hue, sat, val
        h, s, v = f #split into variables
        ang = (h*2*np.pi)/180
        x = (np.cos(ang)*(s/2))+127.5 #find the x
        y = (np.sin(ang)*(s/2))+127.5 #find the y
        f = x, y, v #reassign to f
        f = np.rot90(np.array(f, dtype=int), 1) #replace with a column stack i think

        self.colours, self.counts = np.unique(f, return_counts=True, axis=0) #return all individual colours and the amount of times they change

        return self.colours, self.counts

    def generatepallet(self, n, r): #n = number of colours in the pallet, r = radius of the blocking

        def generatenames(pallet):

            with open(resource_path("colours.csv"), "r") as f:
                r = csv.reader(f)
                names = []
                colours = []
                for colour in r:
                    names.append(colour[1])
                    colours.append(colour[3:])

            colours = np.array([colours], dtype="uint8")
            h, s, v = cv.cvtColor(colours, cv.COLOR_RGB2HSV)[0].T

            ang = (h*2*np.pi)/180
            x = (np.cos(ang)*(s/2))+127.5 #find the x
            y = (np.sin(ang)*(s/2))+127.5 #find the y
            colours = np.column_stack([x, y, v])

            npal = []
            for colour in pallet:
                deltas = colours - colour
                distances = np.linalg.norm(deltas, ord=2, axis=1.)
                i = np.argmin(distances)
                npal.append(names[i]) 

            return npal

        colours = self.colours
        counts = np.copy(self.counts)
        if colours is None:
            colours, counts = self.generatecolours()
        

        pallet = np.array([[0,0,0] for _ in range(n)], dtype=int)
        start = time.time()
        cnum = 0

        while cnum < n: #while theres not enough colours

            i = np.argmax(counts) #get the index of the most common colour
            c = colours[i]
            pallet[cnum] = c #put the colour in the pallet at cnum
            counts[i] = 0 #zeros the count preventing reselection unless we run out of colours

            if cnum > 0:
                distances = np.sort(np.linalg.norm(pallet - c, ord=2, axis=1.)) #calculate the distances in accending order
                dis = distances[np.where(distances > 0.)[0][0]] #calculate the lowest distance that isnt zero
                if dis >= r: cnum += 1  #work out if its too close
            else: cnum += 1 #make the c num of the pallet the current colour

            if time.time()-start >= 3:
                print("Colour error -> reduce blanking radius")
                break

        pallet = np.array(pallet, dtype="uint8") #might be problematic or just unneccicary
        self.pallet = pallet #assigns to the class
        self.npal = generatenames(pallet)

        return pallet #also returns it for fun

    def generateimage(self, w, h, pallet=None):
            
        #variables
        if pallet is None: pallet = self.pallet #if we dont have a pallet take the class pallet
        self.w = w
        self.h = h
        p = cv.resize(self.f, (h, w), interpolation=cv.INTER_LINEAR)
        bgr = XYV2BGR(pallet)

        what = np.copy(p) #you might need to do a deepcopy of this -> easy fix dont worry

        
        what = np.reshape(what, (w*h,3))
        what = np.rot90(what, -1) #make what 3 big long lists -> hue, sat, val
        h, s, v = what
        ang = (h*2*np.pi)/180
        x = (np.cos(ang)*(s/2))+127.5 #find the x
        y = (np.sin(ang)*(s/2))+127.5 #find the y
        what = x, y, v
        what = np.rot90(what) #rotate it back -> probably a better way to do this to be honest

        indexes = np.zeros(len(what)) #makes a zeros array for the length of what
        
        for i in range(len(what)):
            distances = np.linalg.norm(pallet - what[i], ord=2, axis=1.)
            c = np.argmin(distances) #gets the index of the closest pallet colour

            what[i] = bgr[c] #assigns the current pixle to the pallet colour closest
            indexes[i] = c

        image = np.reshape(what, (p.shape))
        indexes = np.reshape(indexes, (p.shape[:2]))


        self.i = image
        self.indexes = indexes
        return image

    def generategrid(self):

        a = np.array(self.indexes, dtype=int)
        p = np.array(XYV2BGR(self.pallet), dtype=int)

        #variable setup
        spc = 50 #size per cell (px)
        fgcolour = (0,0,0)
        textt = 3 #text thickness
        texts = 4 #text size
        t = 1 #line thickness (px)
        f = cv.FONT_HERSHEY_PLAIN

        #calculating dimentions
        columns, rows = self.h, self.w
        gridx, gridy = (columns + 2) * spc, (rows + 2) * spc
        
        #generating blank image
        img = np.zeros((gridy, gridx,3), np.uint8)
        img.fill(255)

        #draw verticle lines
        for x in range(columns+1):
            xval = (x * spc) + spc
            cv.line(img,(xval, spc),(xval, gridy-spc),fgcolour,t)
            
        for x in range(columns): #setup text draw
            xval = (x * spc) + spc
            for y in range(rows):
                yval = (y * spc) + (2 * spc)
                c = p[a[y][x]] #get colour from pallet
                fg = ( int (c [ 0 ]), int (c [ 1 ]), int (c [ 2 ]))
                cv.putText(img, str(a[y][x]+1), (xval+5, yval), f,texts,fg,textt,cv.LINE_4) #draw colour number
        
        #draw horizontal lines
        for y in range(rows+1):
            yval = (y * spc) + spc
            cv.line(img,(spc, yval),(gridx-spc, yval),fgcolour,t)

        return img

    def generatelineby(self):

        imagearray = self.indexes
        npal = self.npal

        with open("line by line.txt", "w") as f:

            for i in range(len(imagearray)):
                f.write(f"Line {i+1}:\n")
                if i % 2 == 0: row = imagearray[i]
                else:
                    row = np.flip(imagearray[i])
                    

                stitchlist = [[-1,0]] #format: [colour code, number of stitches]
                for x in row:
                    if x == stitchlist[-1][0]: stitchlist[-1][1] += 1
                    else: stitchlist.append([x, 1])
                stitchlist.pop(0)

                for ins in stitchlist:
                    i = int(ins[0])
                    f.write(f"    {ins[1]} stitches of {npal[i]}\n")

    def generateoutputs(self):
        pallet = self.pallet
        
        cv.imwrite("output.jpg", cv.resize(self.i, (self.f.shape[1]//2, self.f.shape[0]//2), interpolation=cv.INTER_NEAREST))
        pallet = cv.resize(np.array([XYV2BGR(pallet)]), (720,720), interpolation=cv.INTER_NEAREST)
        cv.imwrite("pallet.jpg", pallet)
        cv.imwrite("grid.jpg", self.generategrid())
        self.generatelineby()

class gui():
    def __init__(self):
        name = "Crochet generator (0.2.2)"

        colour = "#" + hex(randint(0,255))[3:] + hex(randint(0,255))[3:] + hex(randint(0,255))[3:]

        self.output = False
        self.pout = False
        self.image = None
        self.root = root = tk.Tk()
        root.title(name)
        root.resizable(0, 0)
        root.configure(bg=colour)

        titlefont = Font(family="Vivaldi", size=35, slant="italic") #fun fonts for the title and important text
        textfont = Font(family="Vivaldi", size=15, slant="italic")

        self.settings = settingsframe = tk.Frame(root, relief="raised", bd=2)
        #>
        title = tk.Label(settingsframe, text=name, font=titlefont)
        description = tk.Label(settingsframe, text="(c) William Greenwood 2022", font=textfont)
        importb = tk.Button(settingsframe, command=self.importimage, text="Import image")
        self.stitches = stitches = tk.Label(settingsframe, text="Columns: 1, Stitches: 1", font=textfont)
        self.rowscale = rowscale = tk.Scale(settingsframe, length=400, orient="horizontal", from_=1, to=200, resolution=1, label="Rows")
        self.ascale = ascale = tk.Scale(settingsframe, length=400, orient="horizontal", from_=0.1, to=3, resolution=0.1, label="Aspect")
        self.cscale = cscale = tk.Scale(settingsframe, length=400, orient="horizontal", from_=1, to=9, resolution=1, label="Colour number")
        self.rscale = rscale = tk.Scale(settingsframe, length=400, orient="horizontal", from_=1, to=200, resolution=1, label="Blanking radius")
        gen = tk.Button(settingsframe, text="Generate", font=textfont, command=self.outputimage)
        rowscale.set(50)
        ascale.set(0.9)
        cscale.set(5)
        rscale.set(40)
        #~
        title.pack()
        description.pack()
        importb.pack()
        stitches.pack()
        rowscale.pack()
        ascale.pack()
        cscale.pack()
        rscale.pack()
        gen.pack()
        #<
        settingsframe.grid(row=0, column=0, padx=10, pady=10)
        #~
        self.panel = panel = tk.Label(root, relief="raised", bd=2)
        #~
        panel.grid(row=0, column=1, padx=10, pady=10)

    def importimage(self): #get the image path and set the image to self.image
        print("    Importing...")
        path = filedialog.askopenfilename(initialdir = "/",title = "Select a File please")
        print(f"    {path} Selected...")

        image = cv.imread(path)
        if image is not None:
            self.image = image

    def outputimage(self):
        self.output = True

    def getvalues(self): #get slider values and return them for the processing thread
        rows = self.rowscale.get() #setup all variables
        aspect = self.ascale.get()
        colournumber = self.cscale.get()
        radius = self.rscale.get()
        if self.pout == True:
            self.output = False
        self.pout = self.output

        return [self.image, self.output], [rows, aspect], [colournumber, radius]

    def updateimg(self, fullimage, image, cols, stitches): #update function called from the processing thread

        height = 426

        width = round((height/fullimage.shape[0]) * fullimage.shape[1])
        image = cv.resize(image, (width, height), interpolation=cv.INTER_NEAREST)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #convert to rgb image

        image = Image.fromarray(np.uint8(image)).convert('RGB') #make a PIL image
        image = ImageTk.PhotoImage(image) #make a tkinter image

        self.stitches.configure(text=f"Columns: {cols}, Stitches: {stitches}")
        self.panel.configure(image=image)
        self.panel.image = image

def XYV2BGR(a): #takes an array of XYV colours and outputs a BGR array

    a = np.array(a, dtype=int)

    #dont totally trust this bit -> got the hue by trial and error becasue following the docs didnt work

    xy = a[:,:2] #get a list of the x, y coords (dodgey as fuck)
    a = np.rot90(a, -1) #rotate the rest to make it a list of x, y, v
    x, y, v = a

    h = (np.arctan2(y-128, x-128)+np.pi)
    h = ((h + np.pi) % (2 * np.pi))*(90/np.pi)
    s = np.linalg.norm(xy-128, axis=1)*2 #100% a dimension error for this one

    h[h>180] = 180
    s[s>255] = 255
    v[v>255] = 255

    a = np.column_stack([h, s[::-1], v])[::-1]

    a = np.array([a], dtype="uint8")

    a = cv.cvtColor(a, cv.COLOR_HSV2BGR)[0]

    return a

def monitor():

    loopcounter = 1

    rows, a, c, r = 1, 1, 5, 40
    pimage = None

    while True:
        update = False #dont automaticaly update the gui

        lock.acquire()
        [image, out], recolour, redraw = main.getvalues()
        lock.release()

        if image is not None:

            if np.array_equal(image, pimage): pass
            else:
                f = frame(image)
                [rows, a], [c, r] = recolour, redraw #setup all variables
                cols = round((image.shape[0] * rows) / (image.shape[1] * a))
                f.generatepallet(c, r) #generate pallet with c and r
                f.generateimage(cols, rows) #generate image with the scaling parrameters
                update = True
            pimage = image

            if [rows, a] != recolour or [c, r] != redraw: #see if we need to recolour the image

                [rows, a], [c, r] = recolour, redraw #setup all variables
                cols = round((image.shape[0] * rows) / (image.shape[1] * a))
                f.generatepallet(c, r) #generate pallet with c and r
                f.generateimage(cols, rows) #generate image with the scaling parrameters
                update = True
            
            if out == True:
                print("outputting...")
                f.generateoutputs()

            if update == True:
                fimage = np.array(f.i, dtype="uint8")
                
                lock.acquire()
                main.updateimg(image, fimage, cols, f.indexes.size)
                lock.release()

def resource_path(relative_path): # idk honestly
    if asexe == True:
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)
    else:
        return relative_path

if __name__ == "__main__":

    asexe = False

    main = gui()

    lock = threading.Lock()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()

    main.root.mainloop()

'''
 - make it faster
 - make gui more detailed
 - adapt for video
 - make a fun icon
'''