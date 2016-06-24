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
	"fmt"
	"math"
	"runtime"
	"sync"

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

type action int

type predictionAction struct {
	p      Predictor
	fitter int
	fold   int
}

// Returns alpha per fold then per fitter
type AlphaComputer interface {
	ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, fitterEvs [][]float64) [][]float64
}

type Result struct {
	EV  float64
	Std float64 // Standard deviation
}

/*
// MakeFuncPredictions makes predictions to the function values.
// Returned is ...
func MakeFuncPredictions(x mat64.Matrix, f []float64, fitters []Fitter, folds []Fold) (predictor) {

}
*/

// Estimate estimates the expected value of the function with the given inputs.
// Something about knownP bool for fitting the distribution.
func Estimate(d Distribution, x mat64.Matrix, f []float64, fitters []Fitter, folds []Fold, settings *Settings) float64 {
	// Validate input.
	nSamples, dim := x.Dims()
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
	nFitters := len(fitters)

	// Construct data storage.
	// Prediction for each fitter

	needIndividualEVs := true

	evsFull := make([]float64, nFitters)

	evs := make([][]float64, nFitters)
	for i := range evs {
		evs[i] = make([]float64, nFolds)
	}

	concurrent := settings.Concurrent
	if concurrent == 0 {
		concurrent = runtime.GOMAXPROCS(0)
	}

	// TODO(btracey): Don't use goroutines if not concurrent.

	// Parallel evaluation:
	// Algorithm: Send a {fitter, fold} combo to be predicted. When predicted,
	// that fit is sent along a channel to be used for making predictions, and
	// used for making expected value computations (if necessary)

	type fitMessage struct {
		fitter, fold int
	}
	fitChan := make(chan fitMessage)

	type predMessage struct {
		pred   Predictor
		fitter int
		fold   int
	}
	evChan := make(chan predMessage)
	predChan := make(chan predMessage)

	// Launch prediction scheduler
	go func() {
		for i := 0; i < nFitters; i++ {
			for j := 0; j < nFolds; j++ {
				fitChan <- fitMessage{fitter: i, fold: j}
			}
		}
		if settings.UpdateFull {
			for i := 0; i < nFitters; i++ {
				fitChan <- fitMessage{fitter: i, fold: -1}
			}
		}
		close(fitChan)
	}()

	// Find all of the training indices
	allTrainMap := make(map[int]struct{})
	for _, fold := range folds {
		for _, v := range fold.Train {
			allTrainMap[v] = struct{}{}
		}
	}
	allTrain := make([]int, 0, len(allTrainMap))
	for key := range allTrainMap {
		allTrain = append(allTrain, key)
	}

	// Launch prediction workers. The worker calls the fit routine on the
	// selected data, and sends the result along the predChan and EV chan.
	var fitWg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		fitWg.Add(1)
		go func() {
			defer fitWg.Done()
			for p := range fitChan {
				if p.fold == -1 {
					// Use the fit to all, and send to the EVs
					pred := fitters[p.fitter].Fit(x, f, d, allTrain)
					evChan <- predMessage{
						pred:   pred,
						fitter: p.fitter,
						fold:   p.fold,
					}
					continue
				}
				pred := fitters[p.fitter].Fit(x, f, d, folds[p.fold].Train)
				message := predMessage{
					pred:   pred,
					fitter: p.fitter,
					fold:   p.fold,
				}
				// Send to EVs if we need the EVs from them all
				if needIndividualEVs {
					evChan <- message
				}
				// Send to make predictions.
				predChan <- message
			}
		}()
	}
	// Launch worker to close the channel when all the predictors have been made
	go func() {
		fitWg.Wait()
		close(predChan)
		close(evChan)
	}()

	// Launch EV workers
	var evWg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		evWg.Add(1)
		go func() {
			defer evWg.Done()
			for p := range evChan {
				var ev float64
				integrable := p.pred.Integrable(d)
				if integrable {
					ev = p.pred.ExpectedValue(d)
				} else {
					if settings.EstimateFitEV == -1 {
						panic("stackmc: distribution not integrable")
					}
					//sampF := float64(len(allTrain)) * settings.EstimateFitEV
					evMult := settings.EstimateFitEV
					if evMult == 0 {
						evMult = defaultEVMultiple
					}
					evSamp := int(float64(len(allTrain)) * evMult)
					if evSamp == 0 {
						panic("evsamp 0")
					}
					evSamples := mat64.NewDense(evSamp, dim, nil)
					d.Sample(evSamples)
					for i := 0; i < evSamp; i++ {
						ev += p.pred.Predict(evSamples.RawRowView(i))
					}
					ev /= float64(evSamp)
				}
				if p.fold == -1 {
					evsFull[p.fitter] = ev
				} else {
					evs[p.fitter][p.fold] = ev
				}
			}
		}()
	}

	type predictionSet struct {
		pred   Predictor
		fitter int
		fold   int
		idxs   []int
	}

	// Find the unique indices per fold that need to be predicted, and store
	// an index to construct them.
	uniqueMaps := make([]map[int]int, nFolds) // Goes from global index in index in unique.
	uniques := make([][]int, nFolds)          // Unique index to global index
	for i := range uniqueMaps {
		uniqueMaps[i] = make(map[int]int)
	}
	// Find all of the unique indexes
	for i := 0; i < nFolds; i++ {
		for _, v := range folds[i].Assess {
			uniqueMaps[i][v] = 0
		}
		for _, v := range folds[i].Update {
			uniqueMaps[i][v] = 0
		}
	}
	// Create an index mapping
	for i, m := range uniqueMaps {
		uniques[i] = make([]int, 0, len(m))
		for key := range m {
			m[key] = len(uniques[i])
			uniques[i] = append(uniques[i], key)
		}
	}

	// Construct a data container to hold the predicted values.
	predictions := make([][][]float64, nFolds)
	for i := range predictions {
		predictions[i] = make([][]float64, nFitters)
		for j := range predictions[i] {
			predictions[i][j] = make([]float64, len(uniques[i]))
			for k := range predictions[i][j] {
				predictions[i][j][k] = math.NaN()
			}
		}
	}

	// Launch the prediction scheduler
	predictChan := make(chan predictionSet)
	go func() {
		for p := range predChan {
			sz := len(uniques[p.fold]) / concurrent
			for i := 0; i < concurrent; i++ {
				var idxs []int
				if i == concurrent-1 {
					idxs = uniques[p.fold][i*sz : len(uniques[p.fold])]
				} else {
					idxs = uniques[p.fold][i*sz : (i+1)*sz]
				}
				predictChan <- predictionSet{
					pred:   p.pred,
					fitter: p.fitter,
					fold:   p.fold,
					idxs:   idxs,
				}
			}
		}
		close(predictChan)
	}()

	// Launch workers for doing the actual predictions.
	var predWg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		predWg.Add(1)
		go func() {
			defer predWg.Done()
			row := make([]float64, dim)
			for set := range predictChan {
				for _, idx := range set.idxs {
					mat64.Row(row, idx, x)
					v := set.pred.Predict(row)
					predictions[set.fold][set.fitter][uniqueMaps[set.fold][idx]] = v
				}
			}
		}()
	}

	// Wait for the predictions to be all finished
	predWg.Wait()
	evWg.Wait()

	// Compute the alphas with which to correct the fits.
	alpha := settings.AlphaComputer.ComputeAlpha(f, predictions, folds, uniqueMaps, evs)

	// Fold evs is \sum_folds alpha * ghat. If update full, then use the expected
	// value from the fit to all of the samples.
	foldEVs := make([]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		var ev float64
		if settings.UpdateFull {
			// Use the EV from the fit to all
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evsFull[j]
			}
		} else {
			// Use the EV from this particular fold
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evs[j][i]
			}
		}
		foldEVs[i] = ev
	}

	/*
		// Combine all of the data into a final expected value and an error in mean.
		ev, std := settings.Combiner.Combine()

		panic("need to fix everything")
		return Result{
			EV:  ev,
			Std: std,
		}
	*/

	// fmt.Println("alpha = ", alpha)

	// Make corrections to the expected values based on the alpha-weighted
	// differences.

	// The EV is the average EV of the folds plus the integral correction based
	// on the predictions at the held-out data.
	corrector := settings.Corrector
	if corrector == nil {
		corrector = AverageHeldOut{}
	}

	corrections := corrector.Correct(alpha, uniqueMaps, predictions, d, x, fitters, f, folds, settings)

	for i, v := range corrections {
		foldEVs[i] += v
	}

	// Really easy way is to do error bars based on this estimate. Better way
	// is to do something more principled and allow error bars to be returned from
	// Correct. Problem is that all of these are very correlated, so f - alpha g
	// is probably a better error estimate.

	return stat.Mean(foldEVs, nil)

	/*
		// Collect all of the held-out predictions into a list. Could also do this
		// per-fold to correct each individual EV.

		foldEVs := make([]float64, nFolds)
		for i := 0; i < nFolds; i++ {
			var ev float64
			if settings.UpdateFull {
				// Use the EV from the fit to all
				for j := 0; j < nFitters; j++ {
					ev += alpha[i][j] * evsFull[j]
				}
			} else {
				// Use the EV from this particular fold
				for j := 0; j < nFitters; j++ {
					ev += alpha[i][j] * evs[j][i]
				}
			}
			z := 1 / float64(len(folds[i].Update))
			for _, idx := range folds[i].Update {
				truth := f[idx]
				//predIdx := uniques[i][uniqueMaps[i][idx]]
				predIdx := uniqueMaps[i][idx]
				for l := 0; l < nFitters; l++ {
					pred := predictions[i][l][predIdx]
					ev += z * (truth - alpha[i][l]*pred)
					//fmt.Println("fold,idx", i, idx, truth, pred)
				}
			}
			foldEVs[i] = ev
		}
		return stat.Mean(foldEVs, nil)
	*/
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
	p := fit.Fit(x, f, d, inds)
	if p.Integrable(d) {
		return p.ExpectedValue(d)
	}
	fmt.Println("not integrable")
	if evMult == -1 {
		panic("stackmc: distribution not integrable")
	}
	var ev float64
	_, dim := x.Dims()
	if evMult == 0 {
		evMult = defaultEVMultiple
	}
	evSamp := int(float64(len(inds)) * evMult)
	evSamples := mat64.NewDense(evSamp, dim, nil)
	d.Sample(evSamples)
	for i := 0; i < evSamp; i++ {
		ev += p.Predict(evSamples.RawRowView(i))
	}
	ev /= float64(evSamp)
	return ev
}
