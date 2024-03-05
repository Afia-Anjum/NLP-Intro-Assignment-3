import nltk
import random
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from nltk.tag import StanfordPOSTagger
from nltk.tag.brill import *
import nltk.tag.brill_trainer as bt

### Takes in a file path, and returns a list of sentences. Each sentence is a list of tuples in the form of (word, tag) ###
def format_data(filePath):
	f = open(filePath)
	txt = f.read()
	f.close()

	tokens = txt.strip().split('\n')
	sentences = []
	sentence = []
	for token in tokens:
		if not token:
			sentences.append(sentence)
			sentence = []
		else:
			t = token.split()
			sentence.append((t[0], t[1]))
	return sentences

### Takes in a list of sentences (of tuples) and returns just a list of sentences (for getting tagged output from tagger) ###
def untagged_sentences(testData):
	sentences = []
	for sentence in testData:
		s = []
		for word in sentence:
			s.append(word[0])
		sentences.append(s)
	return sentences

### Takes in a list of sentence tuples and an optional smoothing parameter to train and return a HMM tagger ###
def train_hmm(trainData, gamma=0.1):
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(trainData, lambda fdist, bins: LidstoneProbDist(fdist, gamma, bins))
	return tagger

def train_brill(trainData, tagger):
    Template._cleartemplates()
    templates = fntbl37()
    tagger = bt.BrillTaggerTrainer(tagger, templates, trace=3)
    tagger = tagger.train(trainData, max_rules=250)
    return tagger

### Takes a trained HMM tagger and a list of untagged sentences and returns a list of sentences of tuples of the form (word, tag) ###
def tagged_results(tagger, sentences):
	results = []
	for s in sentences:
		tags = tagger.tag(s)
		results.append(tags)
	return results

### Takes in a filename, correct tagged sentence tuples and predicted tagged sentences tuples to generate an output .txt file ###
def generate_output(filename, correct, predicted):
	filename += '.txt'
	with open(filename, 'w') as f:
		f.write('word\tcorrect\tpredicted\n')
		for i in range(len(correct)):
			s = correct[i]
			for j in range(len(s)):
				word = s[j][0]
				actual = s[j][1]
				pred = predicted[i][j][1]
				out = [word, actual, pred]
				f.write('\t'.join(out))
				f.write('\n')
			f.write('\n')

def main():

	# Load and format all files
	dataDomain1 = format_data('A3DataCleaned/Domain1Train.txt')
	dataDomain2 = format_data('A3DataCleaned/Domain2Train.txt')
	testDomain1 = format_data('A3DataCleaned/Domain1Test.txt')
	testDomain2 = format_data('A3DataCleaned/Domain2Test.txt')
	trainELL = format_data('A3DataCleaned/ELLTrain.txt')
	testELL = format_data('A3DataCleaned/ELLTest.txt')

	# Train and save taggers
	hmm = [train_hmm(dataDomain1), train_hmm(dataDomain2), train_hmm(trainELL)]
	brill = [train_brill(dataDomain1, hmm[0]), train_brill(dataDomain2, hmm[1]), train_brill(trainELL, hmm[2])]
	taggers = [hmm, brill]

	# Testing and generating output files
	tests = [testDomain1, testDomain2, testELL]
	for i in range(len(tests)):
		test = tests[i]
		untagged = untagged_sentences(test)

		for j in range(len(taggers)):
			tagger = taggers[j]

			for k in range(len(tagger)):
				model = tagger[k]
				filename = 'results_'
				if j == 0:
					filename += 'hmm'
				elif j == 1:
					filename += 'brill'

				filename += 'ELL' if k == 2 else str(k+1)

				if i == 0:
					filename += '_testDomain1'
				elif i == 1:
					filename += '_testDomain2'
				elif i == 2:
					filename += '_testELL'

				if k == 2 and (i == 0 or i == 1):
					continue

				results = tagged_results(model, untagged)
				generate_output(filename, test, results)
				print(filename + ' accuracy: ' + str(model.evaluate(test)))
main()