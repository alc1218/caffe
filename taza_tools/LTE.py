from os import walk
import random

"CREATION OF CIFASHION FILE FOR CAFFE"

mypath = './cifashion'

seq_id = 0

strToSave = ''
strToSave_label = ''

for (dirpath, dirnames, filenames) in walk(mypath):

	try:
		filenames.remove('.DS_Store')
	except ValueError:
		pass

	for idx in range(len(filenames)):

		" FUTUR VERSION: TAKE INTO ACCOUNT TO CHECK .DS_Store"
		#filenames.remove('.DS_Store')

		strToSave+=dirpath[1:] # WE SKIPE THE FIRST CHARACTER '.'
		strToSave+="/"
		strToSave+=filenames[idx]

		strToSave+=" "

		strToSave+=str(seq_id)

		strToSave+="\n"

	if len(filenames) > 0:
		seq_id+=1
		strToSave_label+=dirpath[12:] # WE SKIPE THE FIRST CHARACTER '.'
		strToSave_label+="\n"

file = open('cifashionDB_labelAndClass.txt', 'w')
file.write(strToSave)
file.close()

file = open('cifashionDB_label.txt', 'w')
file.write(strToSave_label)
file.close()



"SHUFFLE AND SPLIT DATASET IN TRAIN (80) AND TEST (20)"

lines = open('cifashionDB_labelAndClass.txt', 'r').readlines()

random.shuffle(lines)

idx = 0

for line in lines:

	if idx < len(lines) * 0.8:

		"SAVE TRAIN"

		open('shuffled_train_cifashionDB.txt','a').writelines(line)

	else:

		"SAVE TEST"

		open('shuffled_test_cifashionDB.txt','a').writelines(line)

	idx+=1
