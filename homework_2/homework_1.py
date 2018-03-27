from robot_cmd_ros import *
from random import randint

#1. Scrivere un programma che, su stage, simuli un robot aspirapolvere. Il robot deve navigare (muovendosi in avanti) fino ad un ostacolo, per poi cambiare direzione con un angolo scelto randomicamente e riprendere la navigazione.
begin()
while(1):
	while(laser_center_distance()<1):
		turn(randint(0,360))
	forward(0.8)
	

end()
