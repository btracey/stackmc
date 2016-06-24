// package stackmc implements the StackMC algorithm of Tracey, Wolpert, and Alonso.
// The StackMC algorithm is described in
//
//  Brendan Tracey, David Wolpert, and Juan J. Alonso.  "Using Supervised
//  Learning to Improve Monte Carlo Integral Estimation", AIAA Journal,
//  Vol. 51, No. 8 (2013), pp. 2015-2023.
//  doi: 10.2514/1.J051655
//
// StackMC estimates the value of an integral from a set of samples. StackMC has
// been shown empirically to have a lower expected squared error than Monte Carlo
// sampling, although there are no guarantees of a lower error for any specific
// set of samples.
//
// The main routines in the package are the Estimate routines.
// Estimate is the standard case where the sample generation probability is known.
// Here, a fit to f(x) is made. Probability distribution is contained within the Fitter.
package stackmc

// TODO(btracey): Import transforms for other variables? I.e. ImportanceSampling()
// which takes in the generation distribution and outputs new function values?

import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

var (
	errLen   = "stackmc: length mismatch"
	errFolds = "stackmc: number of folds mismatch"
)

type Fold struct {
	// Training, Alpha, and Correct are the list of samples used in each of the
	// parts of the StackMC procedure. The outer length is the data used per fold.
	// Inner length is the sample number to be used in that stage. Index refers
	// to a sample row.
	Train  []int // rows in Locations used for fitting
	Assess []int // rows in locations used for setting alpha
	Update []int // rows in locations used to update the data fit
}

// Settings controls the fitting samples.
type Settings struct {
	UpdateFull    bool // EV correction to the full fit instead of individual fit EVs
	Concurrent    int  // Number of concurrent. If 0, defaults to GOMAXPROCS. Goroutines are created no matter what.
	AlphaComputer AlphaComputer

	// EstimateFitEV is the multiplier on the number of samples to use to estimate
	// the fitter if it cannot analytically integrate the given distribution.
	// If 0, defaults to 100. If -1, will panic if not analytically integrable.
	EstimateFitEV float64

	Corrector Corrector
}

// Corrector corrects the EV to each fold based on the predictions at the
// held-out data
type Corrector interface {
	Correct(alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d Distribution, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) []float64
}

const defaultEVMultiple = 100

// A fitter can produce a Predictor based on a given set of samples.
type Fitter interface {
	Fit(x mat64.Matrix, f []float64, d Distribution, inds []int) Predictor
}

// A Predictor can predict the function value at a set of x locations, and
// can estimate the expected value. In the case of Estimate, for example, the
// generating probability data should be embedded within the predictor by the user.
type Predictor interface {
	// Predict estimates the value of the function at the given x location.
	Predict(x []float64) float64
	// Integrable returns whether the predictor can be analytically integrated
	// under the distribution.
	Integrable(d Distribution) bool
	// ExpectedValue computes the expected value under the distribution.
	ExpectedValue(d Distribution) float64
}

// Distribution. Needs to be able to generate a set of values so at least the
// fitter can estimate with Monte Carlo.
type Distribution interface {
	Sample(x *mat64.Dense)
}

// Returns alpha per fold then per fitter
type AlphaComputer interface {
	ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, fitterEvs [][]float64) [][]float64
}

type Result struct {
	EV  float64
	Std float64 // Standard deviation
}

// Estimate estimates the expected value of the function with the given inputs.
// Something about knownP bool for fitting the distribution.
func Estimate(d Distribution, x mat64.Matrix, f []float64, fitters []Fitter, folds []Fold, settings *Settings) float64 {
	checkEstimateInputs(d, x, f, fitters, folds, settings)

	uniques, uniqueMaps, allTrain := findUniqueIndexs(folds)

	_, evAll, _, evs, predictions := trainAndPredict(allTrain, uniques, x, f, folds, fitters, d, settings)

	alpha := settings.AlphaComputer.ComputeAlpha(f, predictions, folds, uniqueMaps, evs)

	nFolds := len(folds)
	nFitters := len(fitters)
	foldEVs := make([]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		var ev float64
		if settings.UpdateFull {
			// Use the EV from the fit to all
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evAll[j]
			}
		} else {
			// Use the EV from this particular fold
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evs[i][j]
			}
		}
		foldEVs[i] = ev
	}

	corrector := settings.Corrector
	if corrector == nil {
		corrector = AverageHeldOut{}
	}

	corrections := corrector.Correct(alpha, uniqueMaps, predictions, d, x, fitters, f, folds, settings)

	for i, v := range corrections {
		foldEVs[i] += v
	}

	return stat.Mean(foldEVs, nil)
}

func checkEstimateInputs(d Distribution, x mat64.Matrix, f []float64, fitters []Fitter, folds []Fold, settings *Settings) {
	nSamples, _ := x.Dims()
	if len(f) != nSamples {
		panic(errLen)
	}
	if settings == nil {
		// TODO(btracey): Default settings
		panic("nil settings not coded")
	}
	if d == nil {
		panic("stackmc: nil Distribution")
	}

	nFolds := len(folds)
	if nFolds == 0 {
		panic("stackmc: no training folds")
	}
}

// MCExpectedValue computes the expected value of the function based on normal
// Monte Carlo sampling using the listed samples.
func MCExpectedValue(f []float64, inds []int) float64 {
	var ev float64
	for _, idx := range inds {
		ev += f[idx]
	}
	return ev / float64(len(inds))
}

// FitExpectedValue computes the expected value of the function based purely on
// a fit to the function using the given samples.
func FitExpectedValue(fit Fitter, d Distribution, x mat64.Matrix, f []float64, inds []int, evMult float64) float64 {
	_, dim := x.Dims()
	pred := fit.Fit(x, f, d, inds)
	return predictorExpectedValue(pred, d, evMult, len(inds), dim)
}
