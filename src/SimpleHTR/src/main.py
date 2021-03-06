import sys
import numpy as np
import os
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'


def train(model, loader):
	"train NN"
	epoch = 0 
	bestCharErrorRate = float('inf') 
	noImprovementSince = 0 
	earlyStopping = 5 
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		
		charErrorRate = validate(model, loader)
		
		
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		# if noImprovementSince >= earlyStopping:
		if epoch>1:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		recognized = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img] * Model.batchSize) 
	recognized = model.inferBatch(batch) 
	print('Recognized:', '"' + recognized[0] + '"') 


def testing():
	imgdirs = os.listdir('../data/')
	decoderType = DecoderType.BestPath
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

	output=''
	for (j,dire) in enumerate(imgdirs):
		imgFiles= os.listdir('../data/'+dire)
		# print(imgFiles)
		for (i,f) in enumerate(imgFiles):
			#fnImg=input("Enter the image path in '../data/_____' format\n")
			#fnImg=FilePaths.fnInfer
			fnImg = '../data/'+str(dire)+'/'+ f
			print('fn=' + fnImg)
			img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
			
			batch = Batch(None, [img] * Model.batchSize) # fill all batch elements with same input image
			recognized = model.inferBatch(batch) # recognize text
			print('Recognized:', '"' + recognized[0] + '"')
			output=output+recognized[0]+'  '
			
		output=output+'\n'
	print(output)

def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
	parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)

	else:
		testing();
		#print(open(FilePaths.fnAccuracy).read())
		#model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
		#infer(model, FilePaths.fnInfer)




if __name__ == '__main__':
	main()

