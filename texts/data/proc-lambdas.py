f = file("lambdas.dat")

for line in f.readlines():
	if len(line) > 0:
		eval(line)

print dir()
