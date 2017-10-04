package stackmc

import "math/rand"

// Fold represents the data samples used in each part of the StackMC process.
// Each index represents a point of data, specifically the index to the global
// x and f values passed into Estimate. Each fold is used to estimate the expected
// value.
type Fold struct {
	// Train are the locations used to train the fitting algorithm for this fold
	// (to construct g).
	Train []int
	// Assess are the locations used to assess the quality of the fit (estimating
	// the α parameter).
	Assess []int
	// Update are locations used to correct the expected value of the fit (the
	// term in the Monte Carlo estimate of f - αg).
	Correct []int
}

// KFold generates a set of StackMC folds generated from K-fold sampling. The
// training data are the held-in data, while Asses and Correct are the same
// held-out data sets. The number of folds is k.
func KFold(samples, folds int) []Fold {
	training, testing := kFoldPartition(samples, folds)
	foldsData := make([]Fold, len(training))
	for i := range foldsData {
		foldsData[i].Train = training[i]
		foldsData[i].Correct = testing[i]
		foldsData[i].Assess = testing[i]
	}
	return foldsData
}

// KFoldInd is like KFold, but uses indpendent sampling between the fitting
// and the assess/update folds.
func KFoldInd(samples, folds int, indAssess bool) []Fold {
	// Generate two independent k-folds, then push the testing and training
	// together.
	tr1, _ := kFoldPartition(samples, folds)
	_, te2 := kFoldPartition(samples, folds)
	_, te3 := kFoldPartition(samples, folds)
	var fold Fold
	foldsData := make([]Fold, len(tr1))
	for j := 0; j < len(tr1); j++ {
		fold.Train = tr1[j]
		fold.Correct = te2[j]
		if indAssess {
			fold.Assess = te3[j]
		} else {
			fold.Assess = te2[j]
		}
		foldsData[j] = fold
	}
	return foldsData
}

// kFoldPartition partitions the data into nFolds for training and testing.
func kFoldPartition(nData int, nFolds int) (training [][]int, testing [][]int) {
	if nFolds < 0 {
		panic("negative number of folds")
	}
	if nData < 0 {
		panic("negative amount of data")
	}
	if nFolds > nData {
		nFolds = nData
	}

	// Get a random permutation of the data samples
	perm := rand.Perm(nData)

	training = make([][]int, nFolds)
	testing = make([][]int, nFolds)

	nSampPerFold := nData / nFolds
	remainder := nData % nFolds

	idx := 0
	for i := 0; i < nFolds; i++ {
		nTestElems := nSampPerFold
		if i < remainder {
			nTestElems += 1
		}
		testing[i] = make([]int, nTestElems)
		copy(testing[i], perm[idx:idx+nTestElems])

		training[i] = make([]int, nData-nTestElems)
		copy(training[i], perm[:idx])
		copy(training[i][idx:], perm[idx+nTestElems:])

		idx += nTestElems
	}
	if idx != nData {
		panic("bad logic")
	}
	return
}
