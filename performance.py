import numpy as np
import matplotlib.pyplot as plt

## Global variables
IMAGE_OUTPUT = "./RESULTS/"

## Helper functions
def dprime(gen_scores, imp_scores):
	x = np.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores))
	y = np.sqrt(np.power(np.std(gen_scores),2) + np.power(np.std(imp_scores),2))
	return x / y
def getEER(far, frr):
	far_minus_frr = 1
	eer = 0
	for i, j in zip(far, frr):
		if abs(i-j) < far_minus_frr:
			eer = abs(i+j) / 2.
			far_minus_frr = abs(i-j)
	return eer
def getScores(gen_scores, imp_scores):
	# Separate into square
	thresholds = np.linspace(0, 1, 200)
	far = []
	frr = []
	tpr = []
	for t in thresholds:
		tp = 0
		fp = 0
		tn = 0
		fn = 0
	   
		for g_s in gen_scores:
			if g_s <= t:
				tp += 1
			else:
				fn += 1
		for i_s in imp_scores:
			if i_s < t:
				fp += 1
			else:
				tn += 1
		far.append(fp / (fp + tn))
		frr.append(fn / (fn + tp))
		tpr.append(tp / (tp + fn))
		
	eer = getEER(far, frr)
	return eer, far, frr, tpr


## Plotting Functions
def plot_scoreDist(gen_scores, imp_scores, name=""):
	# Set name of figure
	name = IMAGE_OUTPUT + "score_dist_" + str(name) + "_.png"

	# Plot
	plt.figure()
	plt.hist(gen_scores, color='green', lw=2, histtype='step', hatch='//', label='Genuine Scores')
	plt.hist(imp_scores, color='red', lw=2, histtype='step', hatch='\\', label='Impostor Scores')
	plt.legend(loc='best')
	dp = dprime(gen_scores, imp_scores)
	plt.xlim([0,1])
	plt.title('Score Distribution (d-prime= %.2f)' % dp)

	# Save image and return
	plt.savefig(name, bbox_inches='tight')
	plt.show()
	return
def plot_det(fold_performance, name=""):
	# Set up name
	name = IMAGE_OUTPUT + "det_" + str(name) + "_.png"

	# Plot
	plt.figure()
	foldnum = 1
	for fp in fold_performance:
		eer = fp[0]
		far = fp[1]
		frr = fp[2]
		plt.plot(far, frr, lw=2, label='Fold: %d, EER: %.2f' % (foldnum, eer))
		foldnum += 1
	plt.plot([0,1], [0,1], lw=1, color='black')
	plt.xlabel('FAR')
	plt.ylabel('FRR')
	plt.legend(loc='best')
	plt.title('DET Curve')

	# Save image and return
	plt.savefig(name, bbox_inches='tight')
	plt.show()
	return
def plot_roc(fold_performance, name=""):
	# Set up name
	name = IMAGE_OUTPUT + "roc_" + str(name) + "_.png"

	# Plot
	plt.figure()
	foldnum = 1
	for fp in fold_performance:
		far = fp[1]
		tpr = fp[3]
		plt.plot(far, tpr, lw=2, label='Fold ' + str(foldnum))
		foldnum += 1
	plt.xlabel('FAR')
	plt.ylabel('TAR')
	plt.legend(loc='best')
	plt.title('ROC Curve')

	# Save Image and return
	plt.savefig(name, bbox_inches='tight')
	plt.show()
	return
