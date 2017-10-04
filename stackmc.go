// package stackmc implements the StackMC algorithm of Tracey, Wolpert, and Alonso.
// The StackMC algorithm is described in
//  Brendan Tracey, David Wolpert, and Juan J. Alonso.  "Using Supervised
//  Learning to Improve Monte Carlo Integral Estimation", AIAA Journal,
//  Vol. 51, No. 8 (2013), pp. 2015-2023.
//  doi: 10.2514/1.J051655
//
// StackMC estimates the expected value of a function given a set of samples.
// It works by using a machine learning algorithm to construct a set of approximations
// to the underlying function, and using those functions as control variates
// to the original function.
//
// StackMC has been shown empirically to nearly always have a lower expected
// squared error than Monte Carlo sampling and using only the fitting function.
// This eliminates the need to choose between using the raw Monte Carlo average
// or the fit. This finding is a statistical average, there no guarantees of a
// lower error for any specific set of samples.
//
// The main routine in this package is Estimate, which estimates the expected
// value of the function given samples. For a simple use of Estimate, see [example].
// Much of the documentation of this package describes how to implement advanced
// routines (such as for research purposes), and will not be needed for typical
// users.
//
// StackMC works as follows:
//  - Partition the data into a number of folds (stackmc.Fold).
//  - Train each of the fitters using the training data (Fold.Training), and compute
//    the expected value for that fold.
//  - Evaluate the quality of the data in the folds using the assessing data
//    (Fold.Assess) to set α for each fold.
//  - Compute the expected value using the Monte Carlo samples, the fold expected
//    values, and the predictions of the fitter on the correcting data (Fold.Correct).
// In the standard StackMC algorithm, the data is partitioned using k-fold
// validation, with the held-out samples are used both for assessing and correcting.
// There is a single α for all of the folds, computed as discussed in the paper.
// The correction step is computed as
//  f̃ = 1/K \sum_{k=1}^K 1/|assess_k| \sum_{x_i ∈ assess_k} f(x_i) - α_k g_k(x_i)
//
// The above described the default procedure, though alternate procedures can be
// used through the AlphaComputer (setting α given the Assess predictions) and
// Combiner (combining the predictions and α to find the final estimate) interfaces.
package stackmc

// TODO(btracey): Import transforms for other variables? I.e. ImportanceSampling()
// which takes in the generation distribution and outputs new function values?

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

var (
	errLen   = "stackmc: length mismatch"
	errFolds = "stackmc: number of folds mismatch"
)

// Settings controls the procedure for computing the expected value given the
// available data.
type Settings struct {
	// AlphaCombiner dictates how to assign an α to each fold given the data in
	// fold.Assses. See the documentation of AlphaComputer for more information.
	AlphaComputer AlphaComputer

	// Combiner takes the expected values from the folds, the function values,
	// and the out-of-sample predictions, and combines them to estimate the
	// expected value of the fit.
	Combiner Combiner

	// PredictFull sets whether the fitters should be trained on all of the training data.
	// This field is only useful when implementing a custom Combiner.
	//
	// If PredictFull is false, then the evAll slice passed to the Combiner will
	// be nil. If PredictFull is true, then each fitter will be trained on all
	// of the training data (the union of the Training indices in the folds),
	// and its expected value will be put in the corresponding index of evAll.
	PredictFull bool

	// KeepFits sets whether the Predictor fit to each fold should be retained
	// in the equivalent FoldPrediction.
	// This field is only useful when implementing a custom AlphaComputer or
	// Combiner.
	//
	// If KeepFits is false, the Predictor will be allowed to be garbage collected,
	// and the Predictors field of FoldPrediction will be nil. If KeepFits is
	// true, instead the predictor will be stored in FoldPrediction and
	KeepFits bool
}

type Result struct {
	EV float64

	// TODO(btracey): Are more returns needed?
}

// Estimate estimates the expected value of the function f under the distribution p
//  f̃ ≈ f̂ = \int_x f(x) p(x) dx.
// with the given {x,f(x)} samples. This estimation procedure is performed with
// the given fitting algorithms, folds, and settings. Please see the package-level
// documentation for an overview.
//
// If folds is nil, then 5-fold cross validation is used to set the folds, and
// if settings is nil, then the default settings are used. If weights is nil,
// the samples are all assumed to have equal weight. Estimate panics if the number
// of rows in x is not equal to len(fs), and if weights is non-nil and
// len(weights) != len(fs).
//
// Estimate also panics if p or fitters is nil. Several options for fitting
// functions can be found in this package.
//
// Note that in the default settings, neither of the methods of p are called,
// so these can be fake implementations if the distribution does not fit them.
// The interface choice is to help custom implementations (for example to
// find the expected value of a fit using sampling), and for clarity that the
// argument is intended to be a distribution.
func Estimate(xs mat.Matrix, fs, weights []float64, p distmv.RandLogProber, fitters []Fitter, folds []Fold, settings *Settings) Result {
	nSamples, _ := xs.Dims()

	if settings == nil {
		settings = &Settings{}
	}

	if folds == nil {
		folds = KFold(nSamples, 5)
	} else {
		if len(folds) == 0 {
			panic("stackmc: no training folds")
		}
	}

	if p == nil {
		panic("stackmc: nil distribution")
	}

	if len(fitters) == 0 {
		panic("stackmc: no fitter specified")
	}

	if len(fs) != nSamples {
		panic("stackmc: len f mismatch")
	}
	if weights != nil && len(weights) != nSamples {
		panic("stackmc: len weights mismatch. must be nil or rows(x)")
	}

	uniquePreds := FindUniqueIdxs(folds, len(fitters))

	evAll, evs, fps := trainAndPredict(uniquePreds, xs, fs, weights, folds, fitters, p, settings.PredictFull, settings.KeepFits)

	alphaComputer := settings.AlphaComputer
	if alphaComputer == nil {
		alphaComputer = SingleAlpha{}
	}

	alpha := alphaComputer.ComputeAlpha(xs, fs, weights, p, folds, fps, evs)

	combiner := settings.Combiner
	if combiner == nil {
		combiner = BasicCombiner{}
	}
	ev := combiner.Combine(xs, fs, weights, p, folds, evAll, evs, alpha, fps)

	return Result{EV: ev}
}

// PredictIndices is a struct for the unique indices that need to be predicted
// per fold (when indices are both in Assess and Correct). Unique contains a
// list of such indices, and ToUniqueIdx is a map from the global index (row in xs)
// to the unique index.
type PredictIndices struct {
	Unique      []int
	ToUniqueIdx map[int]int
}

// FoldPrediction contains the predictions of each fold.
type FoldPrediction struct {
	// Predictors contains the functional predictor for this fold (indexed by fitter).
	// This is only stored if settings.KeepFits is true.
	Predictors []Predictor

	// PredictIndices are the unique indices that needed to be predicted.
	PredictIndices

	// Predictions contains the predictions for all of the fitters at each
	// unique index. The first index is for the fitter, the second index is the
	// unique index.
	Predictions [][]float64
}

// FindUniqueIdxs finds the unique indices per fold that need to be predicted.
// The indices that need to be predicted are the indices that appear in the union
// of the Assess and Correct fields of Fold. This reduces computational cost
// by only predicting the value once at each point.
func FindUniqueIdxs(folds []Fold, nFitter int) (ups []PredictIndices) {
	nFolds := len(folds)
	ups = make([]PredictIndices, len(folds))
	for i := range folds {
		l := len(folds[i].Assess)
		ups[i].Unique = make([]int, 0, l)
		ups[i].ToUniqueIdx = make(map[int]int, l)
	}

	// Find the unique predictive distributions in assess and update.
	for i := 0; i < nFolds; i++ {
		for _, v := range folds[i].Assess {
			if _, ok := ups[i].ToUniqueIdx[v]; ok {
				continue
			}
			ups[i].ToUniqueIdx[v] = len(ups[i].Unique)
			ups[i].Unique = append(ups[i].Unique, v)
		}
		for _, v := range ups[i].ToUniqueIdx {
			if _, ok := ups[i].ToUniqueIdx[v]; ok {
				continue
			}
			ups[i].ToUniqueIdx[v] = len(ups[i].Unique)
			ups[i].Unique = append(ups[i].Unique, v)
		}
	}
	return ups
}

// FindAllTrain finds all of the unique indices containing across the training
// data in each of the folds.
func FindAllTrain(folds []Fold) []int {
	nFolds := len(folds)
	m := make(map[int]struct{})
	for i := 0; i < nFolds; i++ {
		for _, v := range folds[i].Train {
			m[v] = struct{}{}
		}
	}
	allTrain := make([]int, len(m))
	var count int
	for idx := range m {
		allTrain[count] = idx
		count++
	}
	return allTrain
}

// trainAndPredict creates the fitters from each of the testing sets, as well
// as the fit to all of the data if necessary. It then predicts the value
// for all of the unique indices for each fold, and stores the data in the
// respective FoldPredictor.
func trainAndPredict(pis []PredictIndices, xs mat.Matrix, fs, weights []float64,
	folds []Fold, fitters []Fitter, p distmv.RandLogProber, predictFull, keepFits bool) (evAll []float64, evs [][]float64, fps []FoldPrediction) {

	nFolds := len(folds)
	nFit := len(fitters)
	_, dim := xs.Dims()

	// First, allocate the memory for the return arguments.
	evs = make([][]float64, nFolds)
	for i := range folds {
		evs[i] = make([]float64, nFit)
	}

	fps = make([]FoldPrediction, nFolds)
	for i := 0; i < nFolds; i++ {
		fps[i].PredictIndices = pis[i]
		if keepFits {
			fps[i].Predictors = make([]Predictor, nFit)
		}
		predictions := make([][]float64, nFit)
		for j := 0; j < nFit; j++ {
			predictions[j] = make([]float64, len(fps[i].Unique))
			for k := range predictions[j] {
				predictions[j][k] = math.NaN()
			}
		}
		fps[i].Predictions = predictions
	}

	// Train on all the indices if necessary.
	if predictFull {
		evAll = make([]float64, nFit)
		allTrain := FindAllTrain(folds)
		for i := range evAll {
			pred := fitters[i].Fit(xs, fs, weights, allTrain)
			evAll[i] = pred.ExpectedValue(p)
		}
	}

	// Train each fitter individually and make predictions.
	for i := range folds {
		for j := range fitters {
			fold := folds[i]
			fitter := fitters[j]

			pred := fitter.Fit(xs, fs, weights, fold.Train)
			if keepFits {
				fps[i].Predictors[j] = pred
			}
			x := make([]float64, dim)
			for k, v := range fps[i].Unique {
				mat.Row(x, v, xs)
				fps[i].Predictions[j][k] = pred.Predict(x)
			}
			evs[i][j] = pred.ExpectedValue(p)
		}
	}
	return evAll, evs, fps
}
