package stackmc_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/btracey/stackmc"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize/functions"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

func ExampleRosengauss() {
	// This example computes the expected value of the Rosenbrock function
	// with Gaussian uncertainty. This is the test case in Fig. 5 of
	// "Using Supervised Learning to Improve Monte Carlo Integral Estimation"

	dim := 10       // input dimension
	nSamples := 300 // number of Monte Carlo samples
	evTrue := 5205.0 * float64(dim-1)

	// Set the function of interest.
	fun := functions.ExtendedRosenbrock{}

	// Set the distribution of interest.
	rnd := rand.New(rand.NewSource(1))
	mu := make([]float64, dim)
	sigma := mat.NewDiagonal(dim, nil)
	for i := 0; i < dim; i++ {
		sigma.SetSymBand(i, i, 4)
	}
	p, ok := distmv.NewNormal(mu, sigma, rnd)
	if !ok {
		panic("bad covariance matrix")
	}

	// Generate the samples.
	xs := mat.NewDense(nSamples, dim, nil)
	for i := 0; i < nSamples; i++ {
		p.Rand(xs.RawRowView(i))
	}

	// Evaluate the function at those samples.
	fs := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		fs[i] = fun.Func(xs.RawRowView(i))
	}

	// Estimate the expected value.
	fitter := &stackmc.Polynomial{Order: 3}
	fitters := []stackmc.Fitter{fitter}
	result := stackmc.Estimate(xs, fs, nil, p, fitters, nil, nil)
	evSMC := result.EV

	// Compare with Monte Carlo.
	evMC := stat.Mean(fs, nil)

	// Compare with the fit to all of the data.
	inds := make([]int, nSamples)
	for i := range inds {
		inds[i] = i
	}
	pred := fitter.Fit(xs, fs, nil, inds)
	evFit := pred.ExpectedValue(p)

	fmt.Printf("Monte Carlo Error: %0.4v\n", math.Abs(evMC-evTrue))
	fmt.Printf("Fit Error: %0.4v\n", math.Abs(evFit-evTrue))
	fmt.Printf("StackMC Error: %0.4v\n", math.Abs(evSMC-evTrue))
	// Output:
	// Monte Carlo Error: 1381
	// Fit Error: 1979
	// StackMC Error: 290.9
}
