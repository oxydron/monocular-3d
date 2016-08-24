# Algoritmo e código-fonte desenvolvidos por Ben Hur Bahia do Nascimento
# Todos direitos reservados.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Segment(object):
	"""docstring for Segment"""
	def __init__(self):
		super(Segment, self).__init__()
		self.values = list()

	def add(self, value):
		self.values.append(value)
		return self

	def get(self):
		return self.values

	def __repr__(self):
		return self.values.__repr__()

def clean(matrix, size=1):
	#print(matrix.shape)
	h, w = matrix.shape
	output = np.copy(matrix)
	for j in range(size,h-size):
		#print('{}'.format(j/h))
		for i in range(size,w-size):
			output[j,i] = np.max(matrix[j-size:j+size, i-size:i+size])

	return output

def expand_only_zero(matrix, mean_size, fill_size):
	output = np.copy(matrix)
	y_max, x_max = matrix.shape

	for j in range(0, y_max,2):
		print('{}'.format(j/y_max))
		for i in range(0, x_max,2):
			if matrix[j, i] <= 0:
				temp = list()
				xmin = max(i-mean_size, 0)
				xmax = min(i+mean_size, x_max)

				ymin = max(j-mean_size, 0)
				ymax = min(j+mean_size, y_max)

				# encontro valor médio
				#current_dis = 9999
				#current_value = 0
				for jj in range(ymin, ymax):
					for ii in range(xmin, xmax):
						if matrix[jj,ii] > 0:
							temp.append(matrix[jj, ii])
							#thisdistance = distance(i,j,ii,jj)
							#if thisdistance < current_dis:
							#	current_dis = thisdistance
							#	current_value = matrix[jj,ii]


				# existem valores para calcular média?
				if len(temp) > 0:
				#if current_value > 0:
					#current_value = np.mean(temp)
					current_value = np.median(temp)
					xmin = max(i-fill_size, 0)
					xmax = min(i+fill_size, x_max)

					ymin = max(j-fill_size, 0)
					ymax = min(j+fill_size, y_max)

					for jj in range(ymin, ymax):
						for ii in range(xmin, xmax):
							if matrix[jj,ii] == 0:
								output[jj,ii] = current_value

	return output

def apply_mean(matrix, size):
	output = np.copy(matrix)
	y_max, x_max = matrix.shape

	for j in range(y_max):
		print('{}'.format(j/y_max))
		for i in range(x_max):
			xmin = max(i-size, 0)
			xmax = min(i+size, x_max)

			ymin = max(j-size, 0)
			ymax = min(j+size, y_max)
			
			output[j,i] = np.median(matrix[ymin:ymax, xmin:xmax])

	return output

def find_segments(matrix, limit=500):
	h, w = matrix.shape
	segments = list()

	for i in range(h):
		segments.append(list())

	value = matrix[0,0]

	i = 0

	for j in range(h):
		print('{}'.format(j/h))
		# indo para direita
		nextstart = 0
		nextstart_set = False
		value = matrix[j, 0]

		finished = False
		limcounter = 0
		i = 0
		while not finished:
			#print('{}<{}'.format(i+limcounter,w))
			# setando onde iniciar na próxima rodada
			nextstart_set = False
			i = nextstart
			
			finished = i+(limit*0.5) > w

			limcounter = 0
			s = Segment().add(i)

			#print('nextstart {}\nlimcounter {}\n'.format(nextstart, limcounter))

			# percorrendo para direita até o limite
			while i+limcounter < w-1 and limcounter < limit:
				#print('\t{}<{}'.format(i+limcounter,limit))
				limcounter += 1
				# posição inicial + deslocamento
				
				# se essa posição for o valor que procuro
				if matrix[j,i+limcounter] == value:
					# adiciono valor ao segmento
					s.add(i+limcounter)
				else:
					# se o próximo início não tiver sido setado
					if not nextstart_set:
						# seta com a posição atual
						#print('oi')
						nextstart = i+limcounter
						#print(nextstart)
						#print('nextstart {}'.format(nextstart))
						# confirma que a próxima rodada foi definida
						nextstart_set = True
					#else:
					#	finished = True

			segments[j].append(s)
			
			if matrix[j, nextstart] == value:
				for k in range(w):
					if matrix[j, k] != value:
						value = matrix[j, k]
						nextstart = k
			else:
				value = matrix[j, nextstart]
				
			#i += 1


	return segments		

# recebe 2 segmentos
# retorna um mapeamento para cada valor
def mapper(seg1, seg2):
	firstHigher = True
	ratio = 0.0

	if len(seg1) < len(seg2):
		firstHigher = False
		ratio = len(seg1)/len(seg2)
	else:
		ratio = len(seg2)/len(seg1)

	temp = 0.0
	mappings = []

	if firstHigher:
		for seg in seg1:
			mappings.append((seg, seg2[int(temp)]))
			temp += ratio
	else:
		for seg in seg2:
			mappings.append((seg1[int(temp)], seg))
			temp += ratio

	return mappings

def genZ(pairs, h, w):
	result = np.zeros((h,w))

	for j in range(h):
		for lista in pairs[j]:
			for p1, p2 in lista:
				result[j, p1] = p2 - p1

				# temporário, apenas para corrigir os bugs
				if result[j, p1] < 0:
					result[j, p1] = 0
				elif result[j, p1] > 400:
					result[j, p1] = 400

	return result

def main():
	tras = cv2.imread('gray.tras.png',0)
	#obj = cv2.imread('gray.obj.png',0)
	obj = cv2.imread('gray.obj_.png',0)

	h,w = tras.shape

	segments_tras = find_segments(tras)
	segments_obj = find_segments(obj)

	#for i in range(h):
	#	print(len(segments_tras[i]), end=',')
	#print()
	#for i in range(h):
	#	print(len(segments_obj[i]), end=',')

	for i in range(h):
		segments_tras[i] = segments_tras[i][0:10]
		segments_obj[i] = segments_obj[i][0:10]

	pairs = list()
	for i in range(h):
		pairs.append(list())
		for j in  range(9):
			seg1 = segments_tras[i][j].get()
			seg2 = segments_obj[i][j].get()
			pairs[-1].append(mapper(seg1, seg2))

	#print(pairs)
	z = genZ(pairs,h,w)
	z = apply_mean(z, 20)
	#for i in range(2):
	#	z = expand_only_zero(z,2,50)
	# X = np.arange(0, w)
	# Y = np.arange(0, h)
	# X, Y = np.meshgrid(X, Y)
	# #R = np.sqrt(X**2 + Y**2)
	# Z = z
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
	# ax.set_zlim(0, 150)

	# fig.colorbar(surf, shrink=0.5, aspect=5)
	# plt.show()
	
	writecsv(z)

	im = plt.imshow(z, cmap='jet')
	plt.colorbar(im, orientation='horizontal')
	plt.show()
	# procurando menor 

def writecsv(matrix, filename='output.csv'):
	h,w = matrix.shape
	
	with open(filename,'w') as arq:
		for j in range(h):
			for i in range(w-1):
				arq.write('{},'.format(matrix[j,i]))
			arq.write('{}\n'.format(matrix[j,w-1]))

if __name__ == '__main__':
	main()
