from Tkinter import *
from model import Network
import numpy as np
from PIL import Image, ImageTk
import sys
from threading import Thread
import time

class GUI(Tk):

    def __init__(self,name):

        Tk.__init__(self, None)

        self.title('Equilibrium Propagation')
        self.net = Network(name=name,hyperparameters={"batch_size":1})
        self.hidden_sizes = self.net.hyperparameters["hidden_sizes"]
        self.n_layers = len(self.hidden_sizes) + 2

        self.canvas = Canvas(self, width=600, height=(self.n_layers+1)*100)
        self.canvas.pack(side=BOTTOM)

        # INDEX OF TEST EXAMPLE (IN THE TEST SET)
        Label(self, text="image").pack(side=LEFT)
        self.index = StringVar()
        self.index.set("0")
        Entry(self, textvariable=self.index, width=5).pack(side=LEFT)

        self.update_canvas(first_time=True)

        Thread(target = self.run).start()        

    def update_canvas(self, first_time = False):

        units  = [(28,28)]  +[(10,n/10) for n in self.hidden_sizes]+[(1,10)]
        pixels = [(140,140)]+ [(n/2,50) for n in self.hidden_sizes]+[(250,25)]

        arrays = [256*layer.eval().reshape(dimensions) for layer,dimensions in zip(self.net.layers,units)]
        images = [Image.fromarray(array).resize(dimensions) for array,dimensions in zip(arrays,pixels)]
        self.imgTks = [ImageTk.PhotoImage(image) for image in images]

        [energy, cost, _] = self.net.measure()

        if first_time:
            self.img_canvas    = [self.canvas.create_image(400, (self.n_layers-k)*100, image=imgTk) for k,imgTk in enumerate(self.imgTks)]
            self.energy_canvas = self.canvas.create_text( 20, 100, anchor=W, font="Purisa", text="Energy = %.1f" % (energy))
            self.cost_canvas   = self.canvas.create_text( 20, 200, anchor=W, font="Purisa", text="Cost = %.4f"   % (cost))
        else:
            for img_canvas, imgTk in zip(self.img_canvas,self.imgTks):
                self.canvas.itemconfig(img_canvas, image=imgTk)
            self.canvas.itemconfig(self.energy_canvas, text="Energy = %.1f" % (energy))
            self.canvas.itemconfig(self.cost_canvas,   text="Cost = %.4f"   % (cost))

    def run(self):

        while True:

            index = self.index.get() # index of the test example in the test set
            if index.isdigit():
                index = int(index)
            index = (hash(index) % 10000) + 60000
            self.net.change_mini_batch_index(index)

            self.net.free_phase(n_iterations=1, epsilon=np.float32(.1))
            
            self.update_canvas()

            # FREQUENCY OF UPDATES (IN SECONDS)
            time.sleep(.1)

if __name__ == "__main__":

    name = sys.argv[1]
    GUI(name).mainloop()