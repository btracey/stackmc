package helper

import (
	"fmt"
	"log"
	"math"
	"sync"

	"github.com/btracey/stackmc"
	"github.com/gonum/floats"
)

// Returns a log-spaced range of samples
func SampleRange(nSpace, min, max int) []int {
	ints := make([]int, nSpace)
	fs := make([]float64, nSpace)
	sampVec := floats.LogSpan(fs, float64(min), float64(max))
	for i := range sampVec {
		if sampVec[i] > 0.5 {
			ints[i] = int(math.Ceil(sampVec[i]))
		} else {
			ints[i] = int(math.Floor(sampVec[i]))
		}
	}
	fmt.Println(ints)
	return ints
}

// A FoldGenerator generates how the folds look for a specific number of samples
type Generator interface {
	Generate(nSamples int) (stackmc.Controler, []stackmc.Sample)
}

type StandardKFold struct {
	Dist             stackmc.Distribution
	Function         func([]float64) float64
	FitterGenerators []func() stackmc.Fitter
	NumFolds         int
	NumDim           int

	//samples   []stackmc.Sample
	//controler stackmc.Controler
}

func (s *StandardKFold) Generate(nSamples int) (stackmc.Controler, []stackmc.Sample) {
	samples := make([]stackmc.Sample, nSamples)
	for i := 0; i < nSamples; i++ {
		x := make([]float64, s.NumDim)
		s.Dist.Rand(x)
		f := s.Function(x)
		samples[i] = stackmc.Sample{
			Loc: x,
			Fun: f,
		}
	}

	trainingSets, testingSets := stackmc.KFold(nSamples, s.NumFolds)
	folds := make([]stackmc.Fold, s.NumFolds)
	for i := range folds {
		folds[i].Training = trainingSets[i]
		folds[i].Alpha = testingSets[i]
		folds[i].Correction = testingSets[i]
	}

	allData := make([]int, nSamples)
	for i := range allData {
		allData[i] = i
	}

	fitters := make([]stackmc.Fitter, len(s.FitterGenerators))
	for i, gen := range s.FitterGenerators {
		fitters[i] = gen()
	}

	controler := stackmc.Controler{
		Fit:    fitters,
		Folds:  folds,
		AllPts: allData,
	}

	return controler, samples
}

// MonteCarlo runs a number of runs of a controler and collects statistics
func MonteCarlo(generator Generator, nSampleSlice []int, nRuns int) [][]*stackmc.Result {
	wg := &sync.WaitGroup{}
	wg.Add(len(nSampleSlice) * nRuns)
	results := make([][]*stackmc.Result, len(nSampleSlice))
	for i := range nSampleSlice {
		results[i] = make([]*stackmc.Result, nRuns)
		for j := 0; j < nRuns; j++ {
			go func(i, j int) {
				defer wg.Done()
				var err error
				controler, samples := generator.Generate(nSampleSlice[i])
				results[i][j], err = stackmc.Estimate(controler, samples)
				if err != nil {
					log.Fatal(err)
				}
			}(i, j)

		}
	}
	wg.Wait()
	return results
}
