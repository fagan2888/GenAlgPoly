import numpy as np


def by_rows(d):
	s = ""
	ks = sorted(d.keys())
	for k in ks:
		s += "%05.3f "%eval(k)
		for x in d[k]:
			s+= "%05.3f "%x

		s += "\n"
	s += "\n"
	return s

def by_cols(d):
	s = ""
	ks = sorted(d.keys())
	for k in ks:
		 s += "%04.3f "%eval(k) 
	s += "\n"
	n = len( d[ ks[0] ] )

	for i in range(n):
		for k in ks:
			s += "%04.3f "%d[k][i]
		s += "\n"
	s += "\n"
	return s

filename = "lambda.errors.Auto-Mpg.txt"

f = file(filename,"r")

# skip head
f.readline()

data = dict()

for line in f.readlines():
	line = line.replace("\n","").split(" ")
	k = line[2]
	if k in data.keys():
		data[k].append( eval(line[1]) )
	else:
		data[k] = [ eval(line[1]) ]

print by_cols(data)
