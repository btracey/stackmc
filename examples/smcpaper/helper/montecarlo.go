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
func SampleRange(nSpace int, min, max float64) []int {
	ints := make([]int, nSpace)
	fs := make([]float64, nSpace)
	sampVec := floats.LogSpan(fs, min, max)
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
func MonteCarlo(generator Generator, nSampleSlice []int, nRuns int) ([][]*stackmc.Result, error) {
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
	return results, nil
}

type Eim struct {
	MeanSquaredError float64
	ErrorInMean      float64
}

type SmcMse struct {
	Mc      Eim
	StackMc Eim
	Fitters []Eim
}

func ErrorInMean(results [][]*stackmc.Result, trueEV float64) []SmcMse {
	mses := make([]SmcMse, len(results)) // one per nSamples
	wg := &sync.WaitGroup{}
	wg.Add(len(results))
	for i := range mses {
		go func(i int) {
			defer wg.Done()
			result := results[i]

			nSamp := len(result)

			nFitters := len(result[0].ExpValFitAll)
			mcSqDiff := make([]float64, nSamp)
			smcSqDiff := make([]float64, nSamp)
			fitSqDiff := make([][]float64, nFitters)
			for j := range fitSqDiff {
				fitSqDiff[j] = make([]float64, nSamp)
			}
			for j, r := range result {
				mcSqDiff[j] = (r.ExpValMc - trueEV) * (r.ExpValMc - trueEV)
				smcSqDiff[j] = (r.ExpValStackMc - trueEV) * (r.ExpValStackMc - trueEV)
				for k, ev := range r.ExpValFitAll {
					fitSqDiff[k][j] = (ev - trueEV) * (ev - trueEV)
				}
			}

			mcMeanSqErr := mean(mcSqDiff)
			mcEim := eim(mcSqDiff, mcMeanSqErr)
			smcMeanSqErr := mean(smcSqDiff)
			smcEim := eim(smcSqDiff, smcMeanSqErr)

			fitterEim := make([]Eim, nFitters)
			for j := range fitSqDiff {

				meanSqErr := mean(fitSqDiff[j])
				eimFit := eim(fitSqDiff[j], meanSqErr)

				fitterEim[j] = Eim{
					MeanSquaredError: meanSqErr,
					ErrorInMean:      eimFit,
				}
			}

			mses[i] = SmcMse{
				Mc: Eim{
					MeanSquaredError: mcMeanSqErr,
					ErrorInMean:      mcEim,
				},
				StackMc: Eim{
					MeanSquaredError: smcMeanSqErr,
					ErrorInMean:      smcEim,
				},
				Fitters: fitterEim,
			}
		}(i)
	}
	wg.Wait()
	return mses
}

func PrintMses(eims []SmcMse, sampVec []int) {
	for i := range eims {
		fmt.Println()
		fmt.Println("Number of samples: ", sampVec[i])
		fmt.Printf("Mc ave sq error:\t%e\tError in mean:\t%e\n", eims[i].Mc.MeanSquaredError, eims[i].Mc.ErrorInMean)
		fmt.Printf("StackMc ave sq error:\t%e\tError in mean:\t%e\n", eims[i].StackMc.MeanSquaredError, eims[i].StackMc.ErrorInMean)
		fmt.Print("Fitter ave sq error:\t")
		for j := range eims[i].Fitters {
			fmt.Printf("%e\t", eims[i].Fitters[j].MeanSquaredError)
		}
		fmt.Print("\n")
		fmt.Print("Fitter error in mean:\t")
		for j := range eims[i].Fitters {
			fmt.Printf("%e\t", eims[i].Fitters[j].ErrorInMean)
		}
		fmt.Println("\n")

	}
}

func mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	invNumSamp := 1 / float64(len(s))
	for _, v := range s {
		sum += invNumSamp * v
	}
	return sum
}

func variance(s []float64, mean float64) float64 {
	if len(s) < 2 {
		return 0
	}
	var sum float64
	invNumSamp := 1 / float64(len(s)-1) // for unbiased estimator
	for _, v := range s {
		sum += invNumSamp * (v - mean) * (v - mean)
	}
	return sum
}

func eim(s []float64, mean float64) float64 {
	return math.Sqrt(variance(s, mean)) / math.Sqrt(float64(len(s)))
}
