from robot_cmd_ros import *
import math


#2. Scrivere un programma che, su stage, simuli un robot che deve percorrere una circonferenza di raggio R specificato a tempo di esecuzione.

begin()
periodo=20.01
while(1):
	raggio=input("Inserisci il raggio: ")
	v=(2 * raggio * math.pi)/periodo
	setSpeed(v,(2*math.pi*(1/periodo)),periodo,stopend=True)
	wait(1)

end()
