package analyze

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/fold"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// Settings is a struct for using the analysis routine.
type Settings struct {
	// StackMC settings
	SettingsSMC  *stackmc.Settings
	Distribution stackmc.Distribution
	Fitters      []stackmc.Fitter

	// Settings for generating a stackMC run.
	Folder fold.Folder
	Func   func([]float64) float64

	Dim int
}

// Analyze runs an instance of the StackMC algorithm and returns the expected values.
//func Analyze(nSamples, nFolds int, distribution stackmc.Distribution, sampler Sampler, f func([]float64) float64, fitters []stackmc.Fitter, folder Folder)(evSMC, evMC float64, evFit []float64){
func Analyze(nSamples int, settings Settings) (smcEV, mcEV float64, fitEV []float64) {
	dim := settings.Dim
	// Create sample locations.
	xs := mat64.NewDense(nSamples, dim, nil)

	settings.Distribution.Sample(xs)

	// Evaluate the function.
	row := make([]float64, dim)
	fs := make([]float64, nSamples)
	for i := range fs {
		fs[i] = settings.Func(mat64.Row(row, i, xs))
	}

	// Generate the folds
	folds := settings.Folder.Folds(nSamples)

	// Find the expected values based on simple Monte Carlo, each fitter being
	// fit to all individually, and the StackMC estimate. Do these all concurrently.
	allInds := make([]int, nSamples)
	for i := range allInds {
		allInds[i] = i
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		mcEV = stackmc.MCExpectedValue(fs, allInds)
	}()

	fitEV = make([]float64, len(settings.Fitters))
	for i := range fitEV {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			fitEV[i] = stackmc.FitExpectedValue(settings.Fitters[i], settings.Distribution, xs, fs, allInds, settings.SettingsSMC)
		}(i)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		smcEV = stackmc.Estimate(settings.Distribution, xs, fs, settings.Fitters, folds, settings.SettingsSMC)
	}()
	wg.Wait()
	return smcEV, mcEV, fitEV
}

type AverageResult struct {
	Samples   int
	Runs      int
	SmcExpErr float64
	SmcEim    float64
	McExpErr  float64
	McEim     float64
	FitExpErr []float64
	FitEim    []float64
}

// AverageAnalysis finds the average behavior at a certain number of samples
// using the given number of runs.
func AverageAnalysis(truth float64, nRuns, nSamples int, settings Settings) AverageResult {
	smcEvs := make([]float64, nRuns)
	for i := range smcEvs {
		smcEvs[i] = math.NaN()
	}
	mcEvs := make([]float64, nRuns)
	fitEvs := make([][]float64, nRuns)

	nWorkers := runtime.GOMAXPROCS(0)
	id := make(chan int)
	var wg sync.WaitGroup
	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := range id {
				smcEvs[k], mcEvs[k], fitEvs[k] = Analyze(nSamples, settings)
			}
		}()
	}
	for i := 0; i < nRuns; i++ {
		id <- i
	}
	close(id)
	wg.Wait()

	var r AverageResult
	r.SmcExpErr, r.SmcEim = expErrEim(smcEvs, truth)
	r.McExpErr, r.McEim = expErrEim(mcEvs, truth)

	nFitter := len(settings.Fitters)
	r.FitExpErr = make([]float64, nFitter)
	r.FitEim = make([]float64, nFitter)
	fitEv := make([]float64, nRuns)
	for i := range settings.Fitters {
		// Collect the EVs for fitter i
		for j := 0; j < nRuns; j++ {
			fitEv[j] = fitEvs[j][i]
		}
		r.FitExpErr[i], r.FitEim[i] = expErrEim(fitEv, truth)
	}
	r.Samples = nSamples
	r.Runs = nRuns
	return r
}

// expErrEim computes the expected squared error and error in the mean from a
// set of results and the true value.
func expErrEim(a []float64, truth float64) (expErr, eim float64) {
	e := make([]float64, len(a))
	for i, v := range a {
		e[i] = math.Abs(v - truth)
	}
	mean, std := stat.MeanStdDev(e, nil)
	return mean, stat.StdErr(std, float64(len(e)))
}

func SweepAverage(truth float64, nRuns int, sampleVec []int, settings Settings) []AverageResult {
	results := make([]AverageResult, len(sampleVec))
	var wg sync.WaitGroup
	for i, nSamples := range sampleVec {
		wg.Add(1)
		go func(i, nSamples int) {
			defer wg.Done()
			results[i] = AverageAnalysis(truth, nRuns, nSamples, settings)
		}(i, nSamples)
	}
	wg.Wait()
	return results
}

func RoundedLogSpan(sampleVec []int, lb, ub int) {
	span := make([]float64, len(sampleVec))
	floats.LogSpan(span, float64(lb), float64(ub))
	fmt.Println(span)
	for i, v := range span {
		sampleVec[i] = int(floats.Round(v, 0))
	}
	fmt.Println(sampleVec)
	return
}
