from os import walk
from os import listdir
from os.path import isfile, join
import random

mypath = './cifashion'
#mypath = './raw'

"CREATION OF CIFASHION FILE FOR CONTRASTIVE CAFFE LOSS."

strToSave_class1 = ''
strToSave_file1_class1 = ''
strToSave_file2_class1 = ''

strToSave_class0 = ''
strToSave_file1_class0 = ''
strToSave_file2_class0 = ''

for (dirpath, dirnames, filenames) in walk(mypath):

	if '.DS_Store' in filenames:
		filenames.remove('.DS_Store')

	if len(dirnames) == 0:
		"SAME CLASS COMBINATIONS"
		for idx1 in range(len(filenames)):

			idx2 = idx1 + 1
			if idx2 == len(filenames):
				break
			
			while idx2 < len(filenames):

				#dirpath[1:] --> WE SKIPE THE FIRST CHARACTER '.'

				filePath1 = str(dirpath[1:]) + "/" + str(filenames[idx1])
				filePath2 = str(dirpath[1:]) + "/" + str(filenames[idx2])
				label = " 1\n"

				strToSave_class1 += filePath1 + " " + filePath2 + label

				strToSave_file1_class1 += filePath1 + label
				strToSave_file2_class1 += filePath2 + label

				idx2+=1
	'''
	else:
		"NON SAME CLASS COMBINATIONS"
		for idx1_dirname in range(len(dirnames)):
			print "Progress non same class combinations: " + str(float(idx1_dirname) / float(len(dirnames)))

			idx2_dirname = idx1_dirname + 1
			if idx2_dirname == len(dirnames):
				break

			while idx2_dirname < len(dirnames):

				onlyfiles1 = []
				for f1 in listdir(dirpath + "/" + str(dirnames[idx1_dirname])):
					if f1 != '.DS_Store' and isfile(join(dirpath + "/" +  str(dirnames[idx1_dirname]), f1)):
						onlyfiles1.append(f1)
				onlyfiles2 = []
				for f2 in listdir(dirpath + "/" +  str(dirnames[idx2_dirname])):
					if f2 != '.DS_Store' and isfile(join(dirpath + "/" +  str(dirnames[idx2_dirname]), f2)):
						onlyfiles2.append(f2)

				for file1 in onlyfiles1:
					for file2 in onlyfiles2:
						#dirpath[1:] --> WE SKIPE THE FIRST CHARACTER '.'

						filePath1 = str(dirpath[1:]) + "/" +  str(dirnames[idx1_dirname]) + "/" + str(file1)
						filePath2 = str(dirpath[1:]) + "/" +  str(dirnames[idx2_dirname]) + "/" + str(file2)
						label = " 0\n"

						strToSave_class0 += filePath1 + " " + filePath2 + label

						strToSave_file1_class0 += filePath1 + label
						strToSave_file2_class0 += filePath2 + label

				idx2_dirname+=1
	'''

"SAVE DATASET CLASS 1"

file = open('strToSave_class1.txt', 'w')
file.write(strToSave_class1)
file.close()

"SAVE DATASET CLASS 0"

file = open('strToSave_class0.txt', 'w')
file.write(strToSave_class0)
file.close()

'''
"SAVE DATASET FILE 1 CLASS 1"

file = open('strToSave_file1_class1.txt', 'w')
file.write(strToSave_file1_class1)
file.close()

"SAVE DATASET FILE 2 CLASS 1"

file = open('strToSave_file2_class1.txt', 'w')
file.write(strToSave_file2_class1)
file.close()

"SAVE DATASET FILE 1 CLASS 0"

file = open('strToSave_file1_class0.txt', 'w')
file.write(strToSave_file1_class0)
file.close()

"SAVE DATASET FILE 2 CLASS 0"

file = open('strToSave_file2_class0.txt', 'w')
file.write(strToSave_file2_class0)
file.close()
'''