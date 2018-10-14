// package lsq is a simple package for making least-squares fits.
// This package assumes that the functional approximation is
//  f(x) = β_0 * t_0(x) + β_1 * t_1(x) + ... + β_n * t_n(x)
// where the t_i are functions of the input as set by the Termer, and the β_i
// are free parameters that are set by minimizing the least-squares error over
// a set of training samples.
package lsq

import (
	"math"

	"github.com/btracey/btutil"
	"gonum.org/v1/gonum/mat"
)

// Termer is a type that can set the nonlinear functions from a particular input.
// See the package documentation for more information.
type Termer interface {
	// NumTerms returns the number of terms in the least squares fit as a function
	// of the input dimension of x.
	NumTerms(dim int) int
	// Terms computes the terms given the input, and stores them in-place into
	// terms.
	Terms(terms, x []float64)
}

// Coeffs finds the optimal coefficients given the input data and the Termer.
func Coeffs(xs mat.Matrix, fs, weights []float64, inds []int, t Termer) (beta []float64, err error) {
	_, nDim := xs.Dims()

	nTerms := t.NumTerms(nDim)
	A := mat.NewDense(len(inds), nTerms, nil)
	terms := make([]float64, nTerms)
	row := make([]float64, nDim)
	for i, idx := range inds {
		mat.Row(row, idx, xs)
		t.Terms(terms, row)
		A.SetRow(i, terms)
	}

	b := mat.NewVecDense(len(inds), nil)
	for i, idx := range inds {
		b.SetVec(i, fs[idx])
	}

	if weights != nil {
		// If weights is non-nil, need to do weighted least squares.
		// Need to mulitpy both x and f by sqrt(weight)
		for i, idx := range inds {
			sw := math.Sqrt(weights[idx])
			row := A.RawRowView(i)
			for j := range row {
				row[j] *= sw
			}
			v := b.At(i, 0)
			b.SetVec(i, v*sw)
		}
	}

	beta = make([]float64, nTerms)
	betaVec := mat.NewVecDense(len(beta), beta)
	err = betaVec.SolveVec(A, b)
	if err != nil {
		btutil.PrintMat("A", A)
	}
	return beta, err
}
