import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

root = tk.Tk()
root.title('txt')

win = tk.Toplevel()
win.title('img')

# adding a label
aLabel = ttk.Label(win, text="A Label")
aLabel.grid(column=0, row=0)

photo = tk.PhotoImage(file='./face/1.jpg')
img = cv2.imread('./face/1.jpg')
img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
photo2 =  ImageTk.PhotoImage(image=img2)
imageLabel = ttk.Label(win, image=photo2)
imageLabel.grid(column=2, row=0)


def clickMe():
    action.configure(text="** I have been Clicked! **")
    aLabel.configure(foreground='red')
    
# adding a button
action = ttk.Button(win, text="Click Me!", command=clickMe)
action.grid(column=1, row=0)

win.mainloop()
root.mainloop()



