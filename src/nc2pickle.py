from netCDF4 import Dataset
import os 
import pickle
import sys

def get_nc(path, folder, case):
	
	ncfile_u = Dataset(path + folder + case + '_u.nc')
	ncfile_v = Dataset(path + folder + case + '_v.nc')
	ncfile_th = Dataset(path + folder + case + '_th.nc')
	ncfile_qv = Dataset(path + folder + case + '_qv.nc')
	ncfile_sigma = Dataset(path + folder + case + '_sigma.nc')

	u = ncfile_u.variables['u'][:]
	v = ncfile_v.variables['v'][:]
	th = ncfile_th.variables['th'][:]
	qv = ncfile_qv.variables['qv'][:]
	sigma = ncfile_sigma.variables['sigma'][:]

	dict = {'u':u, 'v':v, 'th':th, 'qv': qv,'sigma':sigma}
	with open('../data/pickle/' + case + '.pkl', 'wb') as f:
		pickle.dump(dict, f)
path = '../data/'
folder = '/d32/'
case = 'd32'
get_nc(path, folder, case)

