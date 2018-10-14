package stackmc

import (
	"math"
	"sort"

	"github.com/btracey/stackmc/lsq"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/combin"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"
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
	ExpectedValue(p Distribution) float64
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

	Sampler samplemv.WeightedSampler
	Fitter  Fitter
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
		sampler:   fitmc.Sampler,
		Predictor: pred,
	}, err
}

// PredMCEV computes the expected value of a predictor by using Monte Carlo.
// If a sampler is provided, that will be used to sample from the distribution.
// If no sampler is provided, the input Distribution will be used, in which
// case the Distribution must be a distmv.Rander.
type PredMCEV struct {
	dim     int
	samples int
	sampler samplemv.WeightedSampler
	Predictor
}

func (pred PredMCEV) ExpectedValue(p Distribution) float64 {
	if pred.samples == 0 {
		panic("no samples to estimate expected value")
	}
	if pred.sampler == nil {
		dist := p.(distmv.Rander)
		// Use simple sampling from the distribution.
		var ev float64
		x := make([]float64, pred.dim)
		for i := 0; i < pred.samples; i++ {
			dist.Rand(x)
			ev += pred.Predict(x)
		}
		ev /= float64(pred.samples)
		return ev
	}
	x := mat.NewDense(pred.samples, pred.dim, nil)
	weights := make([]float64, pred.samples)
	pred.sampler.SampleWeighted(x, weights)
	var ev float64
	for i := 0; i < pred.samples; i++ {
		ev += pred.Predict(x.RawRowView(i))
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
	t := PolyTermer{Order: p.Order}
	beta, err := lsq.Coeffs(xs, fs, weights, inds, t)
	if err != nil {
		return nil, err
	}

	pred := &PolyPred{
		Beta:  beta,
		Order: p.Order,
		Dim:   nDim,
	}
	return pred, nil
}

type PolyPred struct {
	Beta  []float64
	Order int
	Dim   int
}

func (p PolyPred) Predict(x []float64) float64 {
	if len(x) != p.Dim {
		panic("fit: length mismatch")
	}
	terms := make([]float64, len(p.Beta))
	PolyTermer{Order: p.Order}.Terms(terms, x)
	return floats.Dot(terms, p.Beta)
}

func (poly PolyPred) ExpectedValue(p Distribution) float64 {
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
		integral := sizeSpace * poly.Beta[0]
		for i := 0; i < poly.Order; i++ {
			for j := 0; j < dim; j++ {
				max := bounds[j].Max
				min := bounds[j].Min
				pow := float64(i + 2) // 1 for order offset and 1 from integral
				inc := poly.Beta[1+i*dim+j] * sizeSpace / (max - min) *
					(1.0 / pow) * (math.Pow(max, pow) - math.Pow(min, pow))
				integral += inc
			}
		}
		return integral / sizeSpace
	case *distmv.Normal:
		dim := t.Dim()
		if poly.Order > 3 {
			panic("Gaussian not programmed with order > 3")
		}
		// The expected value from the constant term
		ev := poly.Beta[0]

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

		for i := 0; i < poly.Order; i++ {
			for j := 0; j < dim; j++ {
				a := poly.Beta[1+i*dim+j]
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

type PolyTermer struct {
	Order int
}

func (p PolyTermer) NumTerms(dim int) int {
	return 1 + p.Order*dim
}

// puts in  1, x_1, x_2, ... x_n , x_1^2, ..., x_n^2, ... , x_1^order, ..., x_n^order
func (p PolyTermer) Terms(terms, x []float64) {
	dim := len(x)
	terms[0] = 1
	for i := 0; i < p.Order; i++ {
		for j, v := range x {
			terms[1+j+dim*i] = math.Pow(v, float64(i)+1)
		}
	}
}

// PolyCross is a Fitter that fits a polynomial to the data with cross-terms.
type PolyCross struct {
	CrossOrder int // Order of the cross-terms.
	IndivOrder int // Order of the individual power terms.
}

// Fit fits a polynomial to the data samples
func (p *PolyCross) Fit(xs mat.Matrix, fs, weights []float64, inds []int) (Predictor, error) {
	_, nDim := xs.Dims()
	t := PolyCrossTermer{CrossOrder: p.CrossOrder, IndivOrder: p.IndivOrder}
	beta, err := lsq.Coeffs(xs, fs, weights, inds, t)
	if err != nil {
		return nil, err
	}

	pred := &PolyCrossPred{
		Beta:       beta,
		CrossOrder: p.CrossOrder,
		IndivOrder: p.IndivOrder,
		Dim:        nDim,
	}
	/*
		pred.Predict(make([]float64, nDim))
		log.Fatal("polycross")
	*/

	return pred, nil
}

// PolyCrossTermer fits the individual powers up to Order and the cross terms.
// Order must be at least 2.
type PolyCrossTermer struct {
	CrossOrder int // Order of the cross-terms.
	IndivOrder int // Order of the individual power terms.

	// TODO(btracey): allow IndivOrder to be 0 by default.
}

func (p PolyCrossTermer) NumTerms(dim int) int {
	// Number of terms is (dim+1)^Cross for all of the cross terms, and then
	// dim*Idiv for the extra terms.
	if p.IndivOrder < p.CrossOrder {
		panic("indiv order must be greater than or equal to CrossOrder")
	}

	// Cross terms.
	k := p.CrossOrder
	terms := combin.Binomial(dim+k, k)
	// Extra individual terms.
	for i := p.CrossOrder; i < p.IndivOrder; i++ {
		terms += dim
	}
	//fmt.Println("terms = ", terms)
	return terms
}

func (p PolyCrossTermer) Terms(terms, x []float64) {
	dim := len(x)
	k := p.CrossOrder
	lens := make([]int, k)
	n := 1
	for i := range lens {
		lens[i] = dim + 1
		n *= dim + 1
	}
	idxs := make([]int, k)
	count := 0
	for i := 0; i < n; i++ {
		v := 1.0
		combin.SubFor(idxs, i, lens)
		// Need to avoid duplicates
		// TODO(btracey): There is definitely a more efficient algorithm here.
		if !sort.IntsAreSorted(idxs) {
			continue
		}
		for _, idx := range idxs {
			if idx == 0 {
				continue
			}
			v *= x[idx-1]
		}
		terms[count] = v
		count++
	}
	for i := p.CrossOrder + 1; i <= p.IndivOrder; i++ {
		for j := 0; j < dim; j++ {
			terms[count] = math.Pow(x[j], float64(i))
			count++
		}
	}
	if count != p.NumTerms(dim) {
		panic("bad number of terms")
	}
	//fmt.Println("poly cross terms is:", p.NumTerms(dim))
	//log.Fatal("check")
	return
}

type PolyCrossPred struct {
	Beta       []float64
	Dim        int
	CrossOrder int // Order of the cross-terms.
	IndivOrder int // Order of the individual power terms.
}

func (p PolyCrossPred) Predict(x []float64) float64 {
	if len(x) != p.Dim {
		panic("fit: length mismatch")
	}
	terms := make([]float64, len(p.Beta))
	PolyCrossTermer{CrossOrder: p.CrossOrder, IndivOrder: p.IndivOrder}.Terms(terms, x)
	return floats.Dot(terms, p.Beta)
}

func (poly PolyCrossPred) ExpectedValue(p Distribution) float64 {
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

		// Set up the iterator.
		k := poly.CrossOrder
		lens := make([]int, k)
		n := 1
		for i := range lens {
			lens[i] = dim + 1
			n *= dim + 1
		}
		idxs := make([]int, k)

		var integral float64
		count := 0
		for i := 0; i < n; i++ {
			combin.SubFor(idxs, i, lens)
			if !sort.IntsAreSorted(idxs) {
				continue
			}
			// Count the indices present and their values
			m := make(map[int]int)
			for _, idx := range idxs {
				m[idx]++
			}
			intTerm := poly.Beta[count] * sizeSpace
			for idx, pow := range m {
				if idx == 0 {
					continue
				}
				max := bounds[idx-1].Max
				min := bounds[idx-1].Min
				intTerm *= (1 / (max - min))
				fp := float64(pow) + 1
				intTerm *= (1.0 / fp) * (math.Pow(max, fp) - math.Pow(min, fp))
			}
			integral += intTerm
			count++
		}
		for i := poly.CrossOrder + 1; i <= poly.IndivOrder; i++ {
			for j := 0; j < dim; j++ {
				max := bounds[j].Max
				min := bounds[j].Min
				fp := float64(i) + 1
				intTerm := poly.Beta[count] * sizeSpace * (1.0 / (max - min)) * (1.0 / fp) * (math.Pow(max, fp) - math.Pow(min, fp))
				integral += intTerm
				count++
			}
		}
		if count != len(poly.Beta) {
			panic("wrong number of terms")
		}
		return integral / sizeSpace
		/*

			count := n
			for i := p.CrossOrder + 1; i <= p.IndivOrder; i++ {
				for j := 0; j < dim; j++ {
					terms[count] = math.Pow(x[j], float64(i))
					count++
				}
			}
			if count != p.NumTerms(dim) {
				panic("bad number of terms")
			}
			return

			// Integral of the constant term.
			integral := sizeSpace * poly.Beta[0]
			if poly.Order == 0 {
				return integral / sizeSpace
			}

			// Integral of the linear term.
			for i := 0; i < dim; i++ {
				max := bounds[i].Max
				min := bounds[i].Min
				integral += poly.Beta[1+i] * sizeSpace / (max - min) *
					0.5 * (max*max - min*min)
			}
			if poly.Order == 1 {
				return integral / sizeSpace
			}

			// Integrate the quadratic term.
			count := dim + 1
			for i := 0; i < dim; i++ {
				max := bounds[i].Max
				min := bounds[i].Min
				integral += poly.Beta[count] * sizeSpace / (max - min) *
					(1.0 / 3) * (max*max*max - min*min*min)
				count++
			}

			// Integrate the cross terms.
			for i := 0; i < dim; i++ {
				maxi := bounds[i].Max
				mini := bounds[i].Min
				for j := i + 1; j < dim; j++ {
					maxj := bounds[j].Max
					minj := bounds[j].Min
					integral += poly.Beta[count] * sizeSpace / ((maxi - mini) * (maxj - minj)) *
						(1.0 / 4) * (maxi*maxi - mini*mini) * (maxj*maxj - minj*minj)
					count++
				}
			}
			if poly.Order == 2 {
				return integral / sizeSpace
			}
			panic("fit: not coded for order > 2")
		*/
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
	_, dim := xs.Dims()
	return FourPred{
		beta:   beta,
		order:  fr.Order,
		dim:    dim,
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

func (fp FourPred) ExpectedValue(p Distribution) float64 {
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
