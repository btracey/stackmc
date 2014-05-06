package stackmc

import (
	"testing"
)

func checkTestingAndTraining(t *testing.T, name string, training, testing [][]int, nSamples, nFolds int) {
	if nFolds > nSamples {
		nFolds = nSamples
	}
	if len(training) != nFolds {
		t.Errorf("Case %s: training does not have %v folds", name, nFolds)
		return
	}
	if len(testing) != nFolds {
		t.Errorf("Case %s: testing does not have %v folds", name, nFolds)
		return
	}

	// Each training sample should be in testing exactly once
	testCount := make([]int, nSamples)
	for _, fold := range testing {
		for _, sample := range fold {
			testCount[sample] += 1
		}
	}
	for _, val := range testCount {
		if val != 1 {
			t.Errorf("Case %s: Testing samples not all there exactly once. Count = %v", name, testCount)
		}
	}

	// All the training samples should be there nFolds - 1 times
	trainCount := make([]int, nSamples)
	for _, fold := range training {
		for _, sample := range fold {
			trainCount[sample] += 1
		}
	}
	for _, val := range trainCount {
		if val != nFolds-1 {
			t.Errorf("Case %s: Training sample count != %v. Count = %v", name, nFolds-1, trainCount)
		}
	}
}

func TestKFold(t *testing.T) {
	// Test that it doesn't panic with even number
	var training, testing [][]int

	for _, test := range []struct {
		nSamples int
		nFolds   int
		Name     string
	}{
		{
			nSamples: 10,
			nFolds:   2,
			Name:     "Even",
		},
		{
			nSamples: 11,
			nFolds:   3,
			Name:     "Uneven",
		},
		{
			nSamples: 24,
			nFolds:   25,
			Name:     "MoreFolds",
		},
		{
			nSamples: 13,
			nFolds:   11,
			Name:     "Slightly more samples",
		},
		{
			nSamples: 13,
			nFolds:   13,
			Name:     "Leave One Out",
		},
	} {
		training, testing = KFold(test.nSamples, test.nFolds)
		checkTestingAndTraining(t, test.Name, training, testing, test.nSamples, test.nFolds)
	}
}
