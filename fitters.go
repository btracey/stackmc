package stackmc

import (
	"math"
	"math/rand"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

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

type Distribution interface {
	Rand(x []float64) // Puts a random location into the receiver
}

// Polynomial fitter in go. Forms a fit to the non-cross terms
type Polynomial struct {
	Order int
	Dist  Distribution // Probability distribution under which the data is generated

	nTerms  int
	polymat *mat64.Dense // matrix of coefficients of the data
	funs    []float64
}

func (p *Polynomial) Set(data []Sample) error {
	// The number of terms in the polynomial fit is the constant term plus
	// the order times the number of dimensions
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

	alphaMat := mat64.Solve(fitmat, f)

	//fmt.Println("alpha mat = ", alphaMat)

	// cblas: 7831.385961833962 2043.3701591373394 776.9295374776391 -1614.7938352643448 -1611.560450493608 850.1557233713121

	alphaVec := make([]float64, p.nTerms)
	alphaMat.Col(alphaVec, 0)

	return &PolyPred{
		alpha:  alphaVec,
		dist:   p.Dist,
		order:  p.Order,
		nTerms: p.nTerms,
	}, nil
}

type PolyPred struct {
	alpha  []float64
	dist   Distribution
	order  int
	nTerms int
}

func (p PolyPred) Predict(x []float64) float64 {
	terms := make([]float64, p.nTerms)
	coeffs(x, terms, p.order)
	return floats.Dot(terms, p.alpha)
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
