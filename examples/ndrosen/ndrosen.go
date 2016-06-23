package main

import (
	"fmt"
	"sync"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/fit"
	"github.com/btracey/stackmc/kfold"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize/functions"
	"github.com/gonum/stat/dist"
)

func main() {
	// This generates the rosen gauss plots
	nDim := 10
	nSampleVec := 8
	sampleVec := make([]float64, nSampleVec)
	floats.LogSpan(sampleVec, 40, 800)
	//floats.LogSpan(sampleVec, 40, 50)
	nRuns := 200
	nFolds := 5

	distribution := fit.Uniform{}
	for i := 0; i < nDim; i++ {
		distribution.Unifs = append(distribution.Unifs, dist.Uniform{Min: -3, Max: 3})
	}
	trueEv := 1924.0 * float64(nDim-1)

	/*
		eseMC := make([]float64, nSampleVec)
		eimMC := make([]float64, nSampleVec)
		eseFit := make([]float64, nSampleVec)
		eseSMC := make([]float64, nSampleVec)
	*/

	stackmcevs := make([][]float64, nSampleVec)
	mcevs := make([][]float64, nSampleVec)
	fitevs := make([][]float64, nSampleVec)
	for i := 0; i < nSampleVec; i++ {
		stackmcevs[i] = make([]float64, nRuns)
		mcevs[i] = make([]float64, nRuns)
		fitevs[i] = make([]float64, nRuns)
	}

	for i, nSamples := range sampleVec {
		for j := 0; j < nRuns; j++ {
			fmt.Println(i, j)
			mcev, fitev, stackmcev := evs(int(nSamples), nDim, nFolds, functions.ExtendedRosenbrock{}.Func, distribution)

			stackmcevs[i][j] = stackmcev
			mcevs[i][j] = mcev
			fitevs[i][j] = fitev

		}
	}

	_ = trueEv
	/*
		fmt.Println(eimMC)
		fmt.Println(eseMC)
		fmt.Println(eseFit)
		fmt.Println(eseSMC)
	*/
}

func evs(nSamples, nDim, nFolds int, function func(x []float64) float64, distribution fit.Distribution) (mcev, fitev, smcev float64) {
	// Generate random samples
	x := mat64.NewDense(nSamples, nDim, nil)
	for i := 0; i < nSamples; i++ {
		distribution.Rand(x.RawRowView(i))
	}

	// Evaluate the function
	fs := make([]float64, nSamples)
	for i := range fs {
		fs[i] = function(x.RawRowView(i))
	}

	fitter := &fit.Polynomial{
		Order: 3,
	}

	/*
		samples := &stackmc.Samples{
			X: x,
			F: fs,
		}
	*/
	allInds := make([]int, nSamples)
	for i := range allInds {
		allInds[i] = i
	}
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		mcev = stackmc.MCExpectedValue(fs, allInds)
	}()
	go func() {
		defer wg.Done()
		fitev = stackmc.FitExpectedValue(fitter, distribution, x, fs, allInds)
	}()
	go func() {
		defer wg.Done()
		training, testing := kfold.Partition(nSamples, nFolds)
		folds := make([]stackmc.Fold, nFolds)
		for i := range folds {
			folds[i].Train = training[i]
			folds[i].Update = testing[i]
			folds[i].Assess = testing[i]
		}
		settings := &stackmc.Settings{UpdateFull: false}
		smcev = stackmc.Estimate(distribution, x, fs, []stackmc.Fitter{fitter}, folds, settings)
	}()
	wg.Wait()
	return mcev, fitev, smcev
}
