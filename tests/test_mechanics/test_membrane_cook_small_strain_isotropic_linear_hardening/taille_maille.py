# fichier python pour tracer la courbe du deplacement au point A de la membrane de Cook en fonction de la taille du maillage
import numpy as np
import matplotlib.pyplot as plt

maillage =np.array([5,10,15,20,25,30,35,40] )
deplacement = np.array([0.0065870, 0.0068524 ,0.00691182,0.00694536,0.00695595,0.00696528,0.00697175,0.00697175])
ordre = ([1,2,3])
depl =([0.00685622,0.00690947,0.00693885])
Deplhigh = ([0.00687410,0.00691581,0.00694350])
#plt.loglog(maillage,deplacement)
plt.plot(ordre,depl)
#plt.xlabel ('Taille du maillage')
#plt.ylabel('deplacement y du point A')

plt.xlabel ('Ordre du maillage')
plt.ylabel('deplacement y du point A')

plt.title('Etude du deplacement du point A en fonction de lordre equal' )
plt.show()