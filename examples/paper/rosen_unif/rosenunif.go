package main

import (
	"math/rand"
	"runtime"
	"time"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/examples/paper/helper"
	"github.com/davecheney/profile"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

func init() {
	mat64.Register(goblas.Blas{})
}

func main() {
	defer profile.Start(profile.CPUProfile).Stop()
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())

	minSamp := 35
	maxSamp := 200
	numSamp := 8

	sampVec := helper.SampleRange(numSamp, minSamp, maxSamp)

	nRuns := 2000

	nDim := 10

	mins := make([]float64, nDim)
	maxs := make([]float64, nDim)
	for i := range maxs {
		mins[i] = -3
		maxs[i] = 3
	}
	dist := stackmc.NewUniform(mins, maxs)

	generator := &helper.StandardKFold{
		Dist:     dist,
		Function: helper.Rosen,
		FitterGenerators: []func() stackmc.Fitter{
			func() stackmc.Fitter {
				return &stackmc.Polynomial{
					Order: 3,
					Dist:  dist,
				}
			},
		},
		NumFolds: 5,
		NumDim:   nDim,
	}

	_ = helper.MonteCarlo(generator, sampVec, nRuns)
}
