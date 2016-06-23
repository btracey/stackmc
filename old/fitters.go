package stackmc

import (
	"math"
	"math/rand"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

type FittableDistribution interface {
	Fit(x [][]float64)
	Distribution
	New() FittableDistribution
}

type IndependentGaussian struct {
	mean []float64
	std  []float64
}

type NoFit struct {
}

func (f NoFit) Rand(x []float64) {}

func (f NoFit) Dim() int {
	return 0
}

func (f NoFit) LogProb(x []float64) float64 {
	return -1
}

func (f NoFit) New() FittableDistribution {
	return NoFit{}
}

func (f NoFit) Fit(x [][]float64) {
	return
}

func NewIndedpendentGaussian(mean, std []float64) *IndependentGaussian {
	if len(mean) != len(std) {
		panic("uneven lengths")
	}
	return &IndependentGaussian{
		mean: mean,
		std:  std,
	}
}

func (ind *IndependentGaussian) New() FittableDistribution {
	return &IndependentGaussian{}
}

func (ind IndependentGaussian) Rand(x []float64) {
	if len(x) != len(ind.mean) {
		panic("wrong length of x")
	}
	for i := range x {
		x[i] = rand.NormFloat64()*ind.std[i] + ind.mean[i]
	}
	return
}

func (i IndependentGaussian) Dim() int {
	return len(i.mean)
}

func (ind *IndependentGaussian) Fit(x [][]float64) {
	dim := len(x[0])
	ind.mean = make([]float64, dim)
	ind.std = make([]float64, dim)
	nSamples := len(x)

	for _, samp := range x {
		for j, val := range samp {
			ind.mean[j] += val
		}
	}
	for i := range ind.mean {
		ind.mean[i] /= float64(nSamples)
	}

	for _, samp := range x {
		for j, val := range samp {
			ind.std[j] += (val - ind.mean[j]) * (val - ind.mean[j])
		}
	}
	for i := range ind.std {
		ind.std[i] /= float64((nSamples - 1))
	}
	return
}

func (ind IndependentGaussian) Mean(x []float64) []float64 {
	if len(x) < len(ind.mean) {
		x = make([]float64, len(ind.mean))
	} else {
		x = x[:len(ind.mean)]
	}
	copy(x, ind.mean)
	return x
}

func (ind IndependentGaussian) Std(x []float64) []float64 {
	if len(x) < len(ind.std) {
		x = make([]float64, len(ind.std))
	} else {
		x = x[:len(ind.std)]
	}
	copy(x, ind.std)
	return x
}

const (
	logRoot2Pi    = 0.91893853320467274178032973640561763986139747363778341281715154048276569592726039769474329863595419762200564662463433744
	negLogRoot2Pi = -logRoot2Pi
)

func (ind IndependentGaussian) LogProb(x []float64) float64 {
	if len(x) != len(ind.mean) {
		panic("wrong size of x")
	}
	// It's a product of the probabilities or the sum of the log probabliities
	var logprob float64
	for i := range ind.mean {
		sigma := ind.std[i]
		mean := ind.mean[i]
		logprob += negLogRoot2Pi - math.Log(sigma) - (x[i]-mean)*(x[i]-mean)/(2*sigma*sigma)
	}
	return logprob
}

type Uniform struct {
	min []float64
	max []float64
}

func NewUniform(min, max []float64) Uniform {
	if len(min) != len(max) {
		panic("uneven lengths")
	}
	return Uniform{
		min: min,
		max: max,
	}
}

func (u Uniform) Rand(x []float64) {
	if len(x) != len(u.min) {
		panic("wrong length of x")
	}
	for i := range x {
		x[i] = rand.Float64()*(u.max[i]-u.min[i]) + u.min[i]
	}
}

func (u Uniform) Dim() int {
	return len(u.min)
}

func (u Uniform) Max(i int) float64 {
	return u.max[i]
}

func (u Uniform) Min(i int) float64 {
	return u.min[i]
}

func (u Uniform) LogProb(x []float64) float64 {
	sizespace := 1.0
	for i, val := range u.min {
		sizespace *= u.max[i] - val
	}
	return -math.Log(sizespace)
}

type Distribution interface {
	Rand(x []float64) // Puts a random location into the receiver
	LogProb(x []float64) float64
}

// Polynomial fitter in go. Forms a fit to the non-cross terms
type Polynomial struct {
	Order int
	Dist  Distribution // Probability distribution under which the data is generated

	nTerms  int
	polymat *mat64.Dense // matrix of coefficients of the data
	funs    []float64

	data    []Sample
	FitDist bool //Fit the probability distribution // TODO: Make this cleaner
}

func (p *Polynomial) Set(data []Sample) error {
	// The number of terms in the polynomial fit is the constant term plus
	// the order times the number of dimensions

	p.data = data
	nDim := len(data[0].Loc)

	nTerms := 1 + p.Order*nDim
	p.nTerms = nTerms

	polymat := mat64.NewDense(len(data), nTerms, nil)

	terms := make([]float64, nTerms)
	for i, samp := range data {
		coeffs(samp.Loc, terms, p.Order)
		polymat.SetRow(i, terms)
	}
	p.polymat = polymat

	//mat64.Format(p.polymat, 0, 0, os.Stdout, 'g')

	funs := make([]float64, len(data))
	for i := range funs {
		funs[i] = data[i].Fun
	}
	p.funs = funs
	return nil
}

func (p *Polynomial) Fit(inds []int) (Predictor, error) {
	//fmt.Println("Fitting with ", len(inds), " inds")
	// Construct a polynomial matrix with all of the inds
	fitmat := mat64.NewDense(len(inds), p.nTerms, nil)
	row := make([]float64, p.nTerms)
	for i, idx := range inds {
		p.polymat.Row(row, idx)
		//fmt.Println("fitmat row i", row)
		fitmat.SetRow(i, row)
	}
	f := mat64.NewDense(len(inds), 1, nil)
	for i, idx := range inds {
		f.Set(i, 0, p.funs[idx])
	}
	//fmt.Println("funs = ", p.funs)
	//fmt.Println("f = ", f)

	// Solve ax = b to get parameters
	//	fmt.Println("fitmat = ", fitmat)
	//	fmt.Println("f = ", f)
	//fitmatRow, fitmatCol := fitmat.Dims()
	//	fmt.Println("Size fitmat", fitmatRow, fitmatCol)
	//frow, fcol := f.Dims()
	//	fmt.Println("Size f ", frow, fcol)

	alphaMat, err := mat64.Solve(fitmat, f)
	if err != nil {
		panic(err)
	}

	//fmt.Println("alpha mat = ", alphaMat)

	// cblas: 7831.385961833962 2043.3701591373394 776.9295374776391 -1614.7938352643448 -1611.560450493608 850.1557233713121

	alphaVec := make([]float64, p.nTerms)
	alphaMat.Col(alphaVec, 0)

	pred := &PolyPred{
		alpha:   alphaVec,
		dist:    p.Dist,
		order:   p.Order,
		nTerms:  p.nTerms,
		fitDist: p.FitDist,

		data: p.data,
		inds: inds,
	}

	//var dist Distribution
	if p.FitDist {
		fit := p.Dist.(interface {
			FittableDistribution
		})
		dist := fit.New()
		vars := make([][]float64, len(inds))
		for i, idx := range inds {
			vars[i] = p.data[idx].Loc
		}
		dist.Fit(vars)

		pred.dist = dist
	}

	return pred, nil
}

type PolyPred struct {
	alpha   []float64
	dist    Distribution
	order   int
	nTerms  int
	fitDist bool

	inds []int
	data []Sample
}

func (p PolyPred) Predict(x []float64) float64 {
	terms := make([]float64, p.nTerms)
	coeffs(x, terms, p.order)
	pred := floats.Dot(terms, p.alpha)
	return pred
	/*
		if !p.fitDist {
			return pred
		}
		logprob := p.dist.LogProb(x)
		return pred * math.Exp(logprob)
	*/
}

func (p PolyPred) EV() float64 {
	switch t := p.dist.(type) {
	default:
		panic("unsuported distribution for Polynomial")
	case Uniform:
		dim := t.Dim()
		sizeSpace := 1.0
		for i := 0; i < dim; i++ {
			sizeSpace *= t.Max(i) - t.Min(i)
		}
		integral := sizeSpace * p.alpha[0]

		//fmt.Println("size ", sizeSpace)
		//fmt.Println("integral ", integral)
		//	fmt.Println("alpha = ", p.alpha)

		for i := 0; i < p.order; i++ {
			for j := 0; j < dim; j++ {
				max := t.Max(j)
				min := t.Min(j)
				pow := float64(i + 2) // 1 for order offset and 1 from integral
				inc := p.alpha[1+i*dim+j] * sizeSpace / (max - min) *
					(1.0 / pow) * (math.Pow(max, pow) - math.Pow(min, pow))

				//				fmt.Println("order = ", i)
				//				fmt.Println("inc = ", inc)
				integral += inc
			}
		}
		//	fmt.Println("EV = ", integral/sizeSpace)
		return integral / sizeSpace
	case *IndependentGaussian:
		dim := t.Dim()
		if p.order > 3 {
			panic("Gaussian not programmed with order > 3")
		}
		// The expected value from the constant term
		ev := p.alpha[0]
		mean := t.Mean(nil)
		std := t.Std(nil)

		for i := 0; i < p.order; i++ {
			for j := 0; j < dim; j++ {
				a := p.alpha[1+i*dim+j]
				m := mean[j]
				s := std[j]
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
	case NoFit:
		// Return the average over all the samples
		var ev float64
		//pred := make([]float64, len(p.data[0].Loc))
		for i, _ := range p.data {
			ev += p.Predict(p.data[i].Loc)
		}
		return ev / float64(len(p.inds))
	}
}

// puts in  1, x_1, x_2, ... x_n , x_1^2, ..., x_n^2, ... , x_1^order, ..., x_n^order
func coeffs(x []float64, coeffvec []float64, order int) {
	//nTerms := 1 + order*len(x)
	dim := len(x)
	coeffvec[0] = 1
	for i := 0; i < order; i++ {
		for j, v := range x {
			coeffvec[1+j+dim*i] = math.Pow(v, float64(i)+1)
		}
	}
}
