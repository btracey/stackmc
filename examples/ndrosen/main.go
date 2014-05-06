package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/btracey/stackmc"
	"github.com/davecheney/profile"
	"github.com/gonum/blas/cblas"
	//"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

// 29282.92193109024 23937.81800515949 23937.81800515949 22813.871966458628
// 29282.92193109024 34104.42469769479 34104.42469769479 41811.925407142946

func init() {
	mat64.Register(cblas.Blas{})
}

func poly(x []float64) float64 {
	var sum float64
	sum += 10.8
	for _, val := range x {
		sum += 2.4 * val
		sum += 3.5 * val * val
		sum += -2.7 * val * val * val
	}
	return sum
}

func rosen(x []float64) float64 {
	if len(x) < 2 {
		panic("must have more than 1 dimension")
	}
	var sum float64
	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) + 100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	return sum
}

func main() {
	defer profile.Start(profile.CPUProfile).Stop()
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())
	nSamples := 100000
	nDim := 2
	nFolds := 5

	mins := make([]float64, nDim)
	maxs := make([]float64, nDim)
	for i := range maxs {
		mins[i] = -3
		maxs[i] = 3
	}
	dist := stackmc.NewUniform(mins, maxs)

	// Generate data randomly
	samples := make([]stackmc.Sample, nSamples)

	for i := range samples {
		loc := make([]float64, nDim)
		dist.Rand(loc)
		samples[i] = stackmc.Sample{
			Loc: loc,
			Fun: rosen(loc),
		}
	}

	trainingSets, testingSets := stackmc.KFold(nSamples, nFolds)

	folds := make([]stackmc.Fold, nFolds)
	for i := range folds {
		folds[i].Training = trainingSets[i]
		folds[i].Alpha = testingSets[i]
		folds[i].Correction = testingSets[i]
	}

	allData := make([]int, nSamples)
	for i := range allData {
		allData[i] = i
	}

	controler := stackmc.Controler{
		Fit: []stackmc.Fitter{
			&stackmc.Polynomial{
				Order: 3,
				Dist:  dist,
			},
		},
		Folds:  folds[:1],
		AllPts: allData,
	}

	result, err := stackmc.Estimate(controler, samples)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
