package walsh

import (
	"math"
	"math/rand"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/lsq"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/combin"
	"gonum.org/v1/gonum/stat/distmv"
)

// Walsh is a Fitter that fits discrete input data. The Walsh fit is essentially
// a discrete Fourier transform to the data
//  f(x) ≈ β_0 + \sum_{i=1}^d β_{1,i} x_i + \sum_{i=1}^d \sum_{j=2}^d  β_{2,i,j} x_i x_j + ...
// The Order field sets the level of interaction. Note that there are a lot of
// cross-terms for higher order interactions. Also note that for higher orders,
// there are only cross terms (no x_i^2), since for bit strings x^n = x for n ≥ 1.
type Fitter struct {
	Order int
}

var _ stackmc.Fitter = &Fitter{}

func (walsh *Fitter) Fit(xs mat.Matrix, fs, weights []float64, inds []int) (stackmc.Predictor, error) {
	_, nDim := xs.Dims()
	// Count the number of coefficients.
	t := termer{Order: walsh.Order}
	beta, err := lsq.Coeffs(xs, fs, weights, inds, t)

	_ = err // intentionally ignore
	/*
		if err != nil {
			return nil, err
		}
	*/

	pred := &Predictor{
		beta:  beta,
		order: walsh.Order,
		dim:   nDim,
	}
	return pred, nil
}

type termer struct {
	Order int
}

func (t termer) NumTerms(dim int) int {
	var nCoeffs int
	for i := 0; i <= t.Order; i++ {
		nCoeffs += combin.Binomial(dim, i)
	}
	return nCoeffs
}

func (w termer) Terms(terms, x []float64) {
	dim := len(x)
	terms[0] = 1
	count := 1
	for order := 1; order <= w.Order; order++ {
		idx := make([]int, order)
		cg := combin.NewCombinationGenerator(dim, order)
		for cg.Next() {
			cg.Combination(idx)
			terms[count] = x[idx[0]]
			for i := 1; i < order; i++ {
				terms[count] *= x[idx[i]]
			}
			count++
		}
	}
	if count != len(terms) {
		panic("walsh: bad count")
	}
}

type Predictor struct {
	beta  []float64
	order int
	dim   int
}

var _ stackmc.Predictor = Predictor{}

func (wp Predictor) Predict(x []float64) float64 {
	if len(x) != wp.dim {
		panic("fit: length mismatch")
	}
	terms := make([]float64, len(wp.beta))
	termer{Order: wp.order}.Terms(terms, x)
	return floats.Dot(terms, wp.beta)
}

func (wp Predictor) ExpectedValue(p distmv.RandLogProber) float64 {
	switch p.(type) {
	default:
		panic("unsuported distribution for Polynomial")
	case *UniformBitString:
		return wp.beta[0]
	}
}

// UniformBitString is a distribution that generates uniform random variables
// over a bit string. The bits here are represented by either 1 or -1.
type UniformBitString struct {
	dim int
	src *rand.Rand
}

func NewUniformBitString(dim int, src *rand.Rand) *UniformBitString {
	return &UniformBitString{dim: dim}
}

var _ distmv.RandLogProber = &UniformBitString{}

func (u *UniformBitString) Dim() int {
	return u.Dim()
}

func (u *UniformBitString) Rand(x []float64) []float64 {
	if x == nil {
		x = make([]float64, u.dim)
	}
	if len(x) != u.dim {
		panic("uniformbitstring: bad input dimension")
	}
	rnd := rand.Intn
	if u.src != nil {
		rnd = u.src.Intn
	}
	for i := 0; i < len(x); i++ {
		v := rnd(2)
		if v == 0 {
			v = -1
		}
		x[i] = float64(v)
	}
	return x
}

func (u *UniformBitString) LogProb(x []float64) float64 {
	// all bit strings have probability 1/2^N
	return -float64(u.dim) * math.Ln2
}
