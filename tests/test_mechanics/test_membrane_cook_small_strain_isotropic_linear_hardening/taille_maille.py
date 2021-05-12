# fichier python pour tracer la courbe du deplacement au point A de la membrane de Cook en fonction de la taille du maillage
import numpy as np
import matplotlib.pyplot as plt

maillage =np.array([0,5,10,15,20,25,30,35,40,45,50,55,60] )
deplacement = np.array([0, 0 ,0,0,0,0,0,0,0,0,0,0,0])
plt.plot (maillage,deplacement)
plt.show()