package stackmc

import (
	"math"

	"github.com/btracey/stackmc/lsq"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// TODO(btracey): A wrapper for non-integrable distribution/fitter combos.

// Fitter is a type that can produce a Predictor based on the samples and weights
// specified by inds. Specifically, all of the available data is passed to Fitter,
// but only the samples specified in inds should be used.
type Fitter interface {
	Fit(xs mat.Matrix, fs, weights []float64, inds []int) (Predictor, error)
}

// A Predictor can predict the function value at a set of x locations, and
// can estimate the expected value.
type Predictor interface {
	// Predict estimates the value of the function at the given x location.
	Predict(x []float64) float64
	// ExpectedValue computes the expected value under the distribution.
	ExpectedValue(p distmv.RandLogProber) float64
}

// FitMCEV wraps a Fitter for when the fit is not analytically integrable under
// the probability distribution. It returns a Predictor who estimates the expected
// value given a number of samples
type FitMCEV struct {
	// Fixed sets the meaning of Samples. If Fixed is false, the Samples value
	// is a multiplier on the number of training samples.
	Fixed bool
	// Number of samples. See Fixed for meaning.
	Samples int

	Fitter Fitter
}

func (fitmc FitMCEV) Fit(xs mat.Matrix, fs, weights []float64, inds []int) (Predictor, error) {
	pred, err := fitmc.Fitter.Fit(xs, fs, weights, inds)
	if pred == nil && err != nil {
		return pred, err
	}
	_, dim := xs.Dims()
	nSamples := fitmc.Samples
	if !fitmc.Fixed {
		nSamples *= len(inds)
	}
	return PredMCEV{
		dim:       dim,
		samples:   nSamples,
		Predictor: pred,
	}, err
}

type PredMCEV struct {
	dim     int
	samples int
	Predictor
}

func (pred PredMCEV) ExpectedValue(p distmv.RandLogProber) float64 {
	if pred.samples == 0 {
		panic("no samples to estimate expected value")
	}
	var ev float64
	x := make([]float64, pred.dim)
	for i := 0; i < pred.samples; i++ {
		p.Rand(x)
		ev += pred.Predict(x)
	}
	ev /= float64(pred.samples)
	return ev
}

// Polynomial is a Fitter that fits a polynomial to the data. The Polynomial fit
// usse all of the individual terms up to order, but none of the cross-terms.
// That is, Polynomial makes a fit
//  f(x) ≈ β_0
//         + β_0,1 * x_0 + β_1,1 * x_1 + ... + β_n,1 * x_n
//         + β_0,2 * x_0^2 + β_1,2 * x_1^2 + ... + β_n,2 * x_n^2
//         + ...
//         + β_0,order * x_0^order + β_1,order * x_1^order + ... + β_n,order * x_n^order
type Polynomial struct {
	Order int
}

// Fit fits a polynomial to the data samples
func (p *Polynomial) Fit(xs mat.Matrix, fs, weights []float64, inds []int) (Predictor, error) {
	_, nDim := xs.Dims()
	t := polyTermer{Order: p.Order}
	beta, err := lsq.Coeffs(xs, fs, weights, inds, t)
	if err != nil {
		return nil, err
	}

	pred := &PolyPred{
		beta:  beta,
		order: p.Order,
		dim:   nDim,
	}
	return pred, nil
}

type PolyPred struct {
	beta  []float64
	order int
	dim   int
}

func (p PolyPred) Predict(x []float64) float64 {
	if len(x) != p.dim {
		panic("fit: length mismatch")
	}
	terms := make([]float64, len(p.beta))
	polyTermer{Order: p.order}.Terms(terms, x)
	return floats.Dot(terms, p.beta)
}

func (poly PolyPred) ExpectedValue(p distmv.RandLogProber) float64 {
	switch t := p.(type) {
	default:
		panic("unsuported distribution for Polynomial")
	case *distmv.Uniform:
		dim := t.Dim()
		bounds := t.Bounds(nil)
		sizeSpace := 1.0
		for i := 0; i < dim; i++ {
			sizeSpace *= bounds[i].Max - bounds[i].Min
		}
		integral := sizeSpace * poly.beta[0]
		for i := 0; i < poly.order; i++ {
			for j := 0; j < dim; j++ {
				max := bounds[j].Max
				min := bounds[j].Min
				pow := float64(i + 2) // 1 for order offset and 1 from integral
				inc := poly.beta[1+i*dim+j] * sizeSpace / (max - min) *
					(1.0 / pow) * (math.Pow(max, pow) - math.Pow(min, pow))
				integral += inc
			}
		}
		return integral / sizeSpace
	case *distmv.Normal:
		dim := t.Dim()
		if poly.order > 3 {
			panic("Gaussian not programmed with order > 3")
		}
		// The expected value from the constant term
		ev := poly.beta[0]

		mu := t.Mean(nil)
		sigma := t.CovarianceMatrix(nil)
		// Check if it is diagonal
		for i := 0; i < sigma.Symmetric(); i++ {
			for j := i + 1; j < sigma.Symmetric(); j++ {
				if sigma.At(i, j) != 0 {
					panic("only coded for diagonal covariance matrices")
				}
			}
		}

		for i := 0; i < poly.order; i++ {
			for j := 0; j < dim; j++ {
				a := poly.beta[1+i*dim+j]
				m := mu[j]
				s := math.Sqrt(sigma.At(j, j))
				switch i + 1 {
				default:
					panic("shouldn't be here")
				case 1:
					ev += a * m
				case 2:
					ev += a * (m*m + s*s)
				case 3:
					ev += a * (m*m*m + 3*m*s*s)
				}
			}
		}
		return ev
	}
}

type polyTermer struct {
	Order int
}

func (p polyTermer) NumTerms(dim int) int {
	return 1 + p.Order*dim
}

// puts in  1, x_1, x_2, ... x_n , x_1^2, ..., x_n^2, ... , x_1^order, ..., x_n^order
func (p polyTermer) Terms(terms, x []float64) {
	dim := len(x)
	terms[0] = 1
	for i := 0; i < p.Order; i++ {
		for j, v := range x {
			terms[1+j+dim*i] = math.Pow(v, float64(i)+1)
		}
	}
}

// Fourier is a fitter that fits Fourier coefficients to data. The fourier fit
// uses all of the terms in both sine and cosine up to order, but none of the
// cross-terms. The Fourier coefficients are spread over the Bounds specified.
type Fourier struct {
	Order  int
	Bounds []distmv.Bound
}

func (fr *Fourier) Fit(xs mat.Matrix, fs, weights []float64, inds []int) (Predictor, error) {
	if weights != nil {
		panic("fourier: not coded for weighted data")
	}
	t := fourierTermer{
		Order:  fr.Order,
		Bounds: fr.Bounds,
	}
	beta, err := lsq.Coeffs(xs, fs, weights, inds, t)
	if err != nil {
		return nil, err
	}
	return FourPred{
		beta:   beta,
		order:  fr.Order,
		bounds: fr.Bounds,
	}, nil
}

type FourPred struct {
	beta []float64
	//dist  Distribution
	order  int
	dim    int
	bounds []distmv.Bound
}

func (fp FourPred) Predict(x []float64) float64 {
	if len(x) != fp.dim {
		panic("fit: length mismatch")
	}
	t := fourierTermer{
		Order:  fp.order,
		Bounds: fp.bounds,
	}
	terms := make([]float64, len(fp.beta))
	t.Terms(terms, x)
	return floats.Dot(terms, fp.beta)
}

func (fp FourPred) ExpectedValue(p distmv.RandLogProber) float64 {
	//switch t := p.dist.(type) {
	switch t := p.(type) {
	default:
		panic("unsuported distribution for Polynomial")
	case *distmv.Uniform:
		// Check that the bounds match the originally set bounds.
		bnds := t.Bounds(nil)
		if len(bnds) != len(fp.bounds) {
			panic("fourier: bound size mismatch")
		}
		for i, v := range fp.bounds {
			if (bnds[i].Min != v.Min) || (bnds[i].Max != v.Max) {
				panic("fourier: bound mismatch")
			}
		}

		// The bounds match. The integral of a Fourier function over the domain
		// is zero, so only the first term matters.
		return fp.beta[0]
	}
}

type fourierTermer struct {
	Order  int
	Bounds []distmv.Bound
}

func (ft fourierTermer) NumTerms(dim int) int {
	return 1 + ft.Order*dim*2
}

func (ft fourierTermer) Terms(terms, x []float64) {
	order := ft.Order
	bounds := ft.Bounds
	dim := len(x)
	// First, set the offset term to 1.
	terms[0] = 1
	for i := 0; i < dim; i++ {
		p := (x[i] - bounds[i].Min) / (bounds[i].Max - bounds[i].Min)
		for j := 0; j < order; j++ {
			sin := math.Sin(2*math.Pi*float64(j+1)*p - math.Pi)
			terms[1+j*dim+i] = sin
		}
	}
	for i := 0; i < dim; i++ {
		p := (x[i] - bounds[i].Min) / (bounds[i].Max - bounds[i].Min)
		for j := 0; j < order; j++ {
			cos := math.Cos(2*math.Pi*float64(j+1)*p - math.Pi)
			terms[1+order*dim+j*dim+i] = cos
		}
	}
}
