#!/usr/bin/env python3

import pandas as pd

def boxp(x,m = ""):
	return "\\boxp{%5.4f}{%5.4f}{%5.4f}{%5.4f}{%5.4f}{white}; %% %s"%(
		x.min(),
		x.quantile(0.25),
		x.quantile(0.50),
		x.quantile(0.75),
		x.max(), m)

def figs_1():
	datafiles = [
		'Abalone_75_lambda.errors.txt',
		'Auto-Mpg_75_lambda.errors.txt',
		'Housing_75_lambda.errors.txt',
		'Kinematics_75_lambda.errors.txt'
	]

	for df in datafiles:
		print(df)
		d = pd.read_csv( df, sep = ' ' )
		g = d.groupby('lambda')

		for key,group in g:
			gk = group['error']
			print(boxp(gk))

		print()

def figs_3():
	datasets = [
		'Abalone',
		'Auto-Mpg',
		'Housing',
		'Kinematics'
	]
	colnames = [
		'ga.eprr.0.7',
		'ga.eprr.0.8',
		'ga.eprr.0.9',
		'ga.epr',
		'lin.reg',
		'svm',
		'rpart',
		'rf',
		'ci.tree',
	]
	for ds_name in datasets:
		f_name = "%s300_results.txt"%ds_name
		ds = pd.read_csv( f_name, sep = ' ')
		print(f_name)
		for col in colnames:
			if not col == 'id':
				print(boxp(ds[col], col))
		print('')



if __name__ == '__main__':
	figs_3()
