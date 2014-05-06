package stackmc

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

/*
type ExpectedValue struct {
	SimpleMonteCarlo float64 // What the expected value is from Simple Monte Carlo
	FitAll           float64 // Expected value from fitting all

}
*/

type Sample struct {
	Loc  []float64 // Locations of the samples, x
	Fun  float64   // Function value at the sample, f(x)
	LogP float64   // The log of the probability of the location, log(p(x))
	LogQ float64   // Generation probability of x, q(x)  (Useful for importance sampling)
}

type Result struct {
	ExpValMc      float64 // Expected value based on the samples for simple monte carlo
	ExpValFitAll  []float64
	ExpValStackMc float64
}

var LengthMismatch error = errors.New("sample length inconsistency")

type Fold struct {
	Training   []int
	Alpha      []int
	Correction []int
}

//
type Controler struct {
	// How many folds into which the data should be split. If this is zero
	// it will be set to 5
	Folds []Fold

	Fit []Fitter

	AllPts []int // Points to be used in the fit to all and computing the MC points
}

// KFold splits the data into K sets. The outputs training and testing each have
// an outer length of nFolds. If nFolds > nData, nFolds will be set to nData.
func KFold(nData int, nFolds int) (training [][]int, testing [][]int) {
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

func checkInputs(samples []Sample) error {
	if len(samples) < 3 {
		return errors.New("stackmc: must have at least 3 samples")
	}

	inputDim := len(samples[0].Loc)
	if inputDim == 0 {
		return errors.New("stackmc: location of the first sample has zero length")
	}

	for i, sample := range samples {
		// Check that all the inputs have the same length
		if len(sample.Loc) != inputDim {
			return fmt.Errorf("stackmc: not all sample locations have the same length. Input Dim is %v, Sample %v has length %v", inputDim, i, len(sample.Loc))
		}
	}
	return nil
}

type predStruct struct {
	Pred   Predictor
	Err    error
	Fold   int
	Fitter int
}

type evStruct struct {
	Ev     float64
	Fold   int
	Fitter int
}

// Estimate estimates the result from MC
// If all
func Estimate(control Controler, samples []Sample) (*Result, error) {

	//fmt.Println("Starting Estimate")
	err := checkInputs(samples)
	if err != nil {
		return nil, err
	}

	nFolds := len(control.Folds)
	nFitter := len(control.Fit)

	for _, fit := range control.Fit {
		fit.Set(samples)
	}

	// There are nFolds per fitting function
	//predictors := make([][]Predictor, nFitter)
	//errs := make([][]error, nFitter)

	// Form all of the predictors in parallel, and then send them along
	// to the next step

	predChan := make(chan predStruct, 1)
	for i := 0; i < nFitter; i++ {
		for j := 0; j < nFolds; j++ {
			go func(i, j int) {
				inds := control.Folds[j].Training
				predictor, err := control.Fit[i].Fit(inds)

				predChan <- predStruct{
					Pred:   predictor,
					Err:    err,
					Fold:   j,
					Fitter: i,
				}

			}(i, j)
		}
	}

	// Launch a goroutine to read form the pred chan, and launch the jobs

	fitEVs := make([][]float64, nFitter)
	for i := range fitEVs {
		fitEVs[i] = make([]float64, nFolds)
	}
	correctionPredictions := make([][][]float64, nFitter)
	alphaPredictions := make([][][]float64, nFitter)
	for i := range correctionPredictions {
		correctionPredictions[i] = make([][]float64, nFolds)
		alphaPredictions[i] = make([][]float64, nFolds)
		for j := range correctionPredictions[i] {
			correctionPredictions[i][j] = make([]float64, len(control.Folds[j].Alpha))
			alphaPredictions[i][j] = make([]float64, len(control.Folds[j].Alpha))
		}
	}

	evWg := &sync.WaitGroup{}
	predWg := &sync.WaitGroup{}
	corrPredWg := &sync.WaitGroup{}
	evWg.Add(nFitter * nFolds)
	predWg.Add(nFitter * nFolds)
	corrPredWg.Add(nFitter * nFolds)
	go func() {
		for i := 0; i < nFitter; i++ {
			for j := 0; j < nFolds; j++ {
				pred := <-predChan
				if err != nil {
					panic(err) // TODO: Fix this
				}

				// Need to do two things. Need to calculate the EV, and need to
				// make predictions at all of the folds.
				go func(pred predStruct) {
					defer evWg.Done()
					ev := pred.Pred.EV()
					fitEVs[pred.Fitter][pred.Fold] = ev
				}(pred)

				go func(pred predStruct) {
					defer predWg.Done()
					inds := control.Folds[pred.Fold].Alpha
					for i, idx := range inds {
						prediction := pred.Pred.Predict(samples[idx].Loc)
						alphaPredictions[pred.Fitter][pred.Fold][i] = prediction
					}
				}(pred)

				go func(pred predStruct) {
					defer corrPredWg.Done()
					inds := control.Folds[pred.Fold].Correction
					for i, idx := range inds {
						prediction := pred.Pred.Predict(samples[idx].Loc)
						correctionPredictions[pred.Fitter][pred.Fold][i] = prediction
					}
				}(pred)
			}
		}
	}()

	// Also launch a goroutine to predict the value at all of the locations
	// and the monte carlo estimate of all the points
	evAll := make([]float64, len(control.Fit))
	allWg := &sync.WaitGroup{}
	allWg.Add(1)
	go func() {
		defer allWg.Done()
		wg := &sync.WaitGroup{}
		wg.Add(len(control.Fit))
		for i, fitter := range control.Fit {
			go func(i int, fit Fitter) {
				defer wg.Done()
				predAll, allErr := fitter.Fit(control.AllPts)
				if allErr != nil {
					panic(allErr) // TODO: Fix
				}
				evAll[i] = predAll.EV()
				/*
					var ev float64
					norm := 1 / float64(nSamples)
					for _, idx := range control.AllPts {
						sample := samples[idx]
						prediction := predAll.Predict(sample.Loc)
						ev += norm * prediction * math.Exp(sample.LogP-sample.LogQ)
					}
					evAll[i] = ev
				*/
			}(i, fitter)
			wg.Wait()
		}
	}()

	// Also compute EV MC
	var evMC float64
	norm := 1 / float64(len(control.AllPts))
	for _, idx := range control.AllPts {
		sample := samples[idx]
		evMC += norm * sample.Fun * math.Exp(sample.LogP-sample.LogQ)
	}

	// Now, we need to wait until all the predictions are done in order to
	// compute alpha
	predWg.Wait()

	// Concatonate all of the predictions together to get the data for alpha
	alphaDatasets := make([][]float64, nFitter+1)
	for _, fold := range control.Folds {
		for _, idx := range fold.Alpha {
			alphaDatasets[0] = append(alphaDatasets[0], samples[idx].Fun)
		}
	}
	for i := 0; i < nFitter; i++ {
		for j := 0; j < nFolds; j++ {
			alphaDatasets[i+1] = append(alphaDatasets[i+1], alphaPredictions[i][j]...)
		}
	}

	covmat, err := cov(alphaDatasets...)
	if err != nil {
		return nil, err
	}

	nAlphaDataset := len(alphaDatasets)

	// The optimal set of alphas is computed by A * alpha = B where
	A := mat64.NewDense(nAlphaDataset-1, nAlphaDataset-1, nil)
	// Covariance of the fitters with one another
	A.Submatrix(covmat, 1, 1, nAlphaDataset-1, nAlphaDataset-1)
	b := mat64.NewDense(nAlphaDataset-1, 1, nil)
	// Covariance of the fitter and f
	b.Submatrix(covmat, 0, 1, nAlphaDataset-1, 1)
	c := mat64.Solve(A, b)

	alpha := make([]float64, nFitter)
	for i := range alpha {
		alpha[i] = c.At(i, 0)
		if math.IsNaN(alpha[i]) {
			fmt.Println(covmat)
			os.Exit(1)
		}
	}

	// Wait for all the expected values to be done
	evWg.Wait()
	corrPredWg.Wait()

	foldEvs := make([]float64, nFolds)
	// fitEV := make([]float64, nFitter)
	// Now, need a corrected expected value. For each fold, find the expected
	// value of f - alpha g
	for i := 0; i < nFolds; i++ {
		var foldEv float64
		norm := 1 / float64(len(control.Folds[i].Correction))
		for j, idx := range control.Folds[i].Correction {
			val := samples[idx].Fun
			for k := 0; k < nFitter; k++ {
				val -= alpha[k] * correctionPredictions[k][i][j]
			}

			foldEv += norm * val
		}
		for k := 0; k < nFitter; k++ {
			foldEv += alpha[k] * fitEVs[k][i]
		}
		foldEvs[i] = foldEv
	}

	var stackMcEv float64
	norm = 1 / float64(nFolds)
	for i := 0; i < nFolds; i++ {
		stackMcEv += norm * foldEvs[i]
	}

	allWg.Wait()

	result := &Result{
		ExpValMc:      evMC,
		ExpValFitAll:  evAll,
		ExpValStackMc: stackMcEv,
	}

	return result, nil
}

func cov(data ...[]float64) (*mat64.Dense, error) {
	nSets := len(data)
	if nSets == 0 {
		return mat64.NewDense(0, 0, nil), nil
	}
	nData := len(data[0])

	for i := range data {
		if len(data[i]) != nData {
			return nil, errors.New("cov: datasets have unequal size")
		}
	}
	covmat := mat64.NewDense(nSets, nSets, nil)

	//fmt.Println("data = ", data)
	//fmt.Println("nData = ", nData)

	// Compute the mean of all the datasets
	means := make([]float64, nSets)
	for i := range means {
		means[i] = floats.Sum(data[i]) / float64(nData)
	}
	//fmt.Println("means = ", means)

	for i := 0; i < nSets; i++ {
		for j := i; j < nSets; j++ {
			var cv float64
			meanI := means[i]
			meanJ := means[j]
			invData := 1 / float64(nData-1)
			for k, val := range data[i] {
				cv += invData * (val - meanI) * (data[j][k] - meanJ)
			}
			covmat.Set(i, j, cv)
			covmat.Set(j, i, cv)
		}
	}
	return covmat, nil
}

type Fitter interface {
	// Set sets the data. This should do any necessary precomputing and scaling
	// etc.
	Set(data []Sample) error

	// Precompute allows the fitting algorithm to transform the data in some way
	// (say rescaling) and compute any necessary other data (say, coefficients
	// of a polynomial matrix)
	//Precompute() error

	// Fit fits the algorithm with the specified indices. This should be able
	// to be called in parallel
	Fit(inds []int) (Predictor, error)
}

// Hmmmm how to deal with probability distributions and fitters? They probably
// need to be coupled.

type Predictor interface {
	Predict(x []float64) float64 // Predicts the value at a given location
	// Gives the expected value given the parameters of the fit (probability
	// distribution is implicit)
	EV() float64
}
