package fit

import (
	"fmt"
	"math"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/distribution"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// Polynomial fitter in go. Forms a fit to the non-cross terms
type Polynomial struct {
	Order int
	//Distribution Distribution // Probability distribution under which the data is generated
}

// Fit fits a polynomial to the data samples
func (p *Polynomial) Fit(x mat64.Matrix, f []float64, inds []int) stackmc.Predictor {
	_, nDim := x.Dims()
	nTerms := 1 + p.Order*nDim
	polymat := mat64.NewDense(len(inds), nTerms, nil)
	terms := make([]float64, nTerms)
	row := make([]float64, nDim)
	for i, idx := range inds {
		mat64.Row(row, idx, x)
		coeffs(terms, row, p.Order)
		polymat.SetRow(i, terms)
	}
	// TODO: Make a better solve when there is a vec.Solve(mat)

	fs := make([]float64, len(inds))
	for i, idx := range inds {
		fs[i] = f[idx]
	}
	fvec := mat64.NewVector(len(fs), fs)

	alpha := make([]float64, nTerms)
	alphaVec := mat64.NewVector(len(alpha), alpha)
	err := alphaVec.SolveVec(polymat, fvec)
	//alphaMat, err := mat64.Solve(polymat, fvec)
	if err != nil {
		fmt.Println("x = ")
		fmt.Printf("%0.4v\n", mat64.Formatted(x))
		fmt.Println("polymat = ")
		fmt.Printf("%0.4v\n", mat64.Formatted(polymat))
		r, c := polymat.Dims()
		fmt.Printf("Polymat dims, r = %v, c = %v\n", r, c)
		panic("error fitting: " + err.Error())
		//return nil, err
	}

	//alphaVec := make([]float64, nTerms)
	//alphaMat.Col(alphaVec, 0)

	pred := &PolyPred{
		alpha: alpha,
		//dist:  p.Distribution,
		order: p.Order,
		dim:   nDim,
	}
	return pred
}

type PolyPred struct {
	alpha []float64
	//dist  Distribution
	order int
	dim   int
}

func (p PolyPred) Predict(x []float64, dist stackmc.Distribution) float64 {
	if len(x) != p.dim {
		panic("fit: length mismatch")
	}
	terms := make([]float64, len(p.alpha))
	coeffs(terms, x, p.order)
	return floats.Dot(terms, p.alpha)
}

func (p PolyPred) Integrable(dist stackmc.Distribution) bool {
	switch dist.(type) {
	case distribution.Uniform, distribution.IndependentGaussian:
		return true
	default:
		return false
	}
}

func (p PolyPred) ExpectedValue(dist stackmc.Distribution) float64 {
	//switch t := p.dist.(type) {
	switch t := dist.(type) {
	default:
		panic("unsuported distribution for Polynomial")
	case distribution.Uniform:
		dim := t.Dim()
		sizeSpace := 1.0
		for i := 0; i < dim; i++ {
			sizeSpace *= t.Unifs[i].Max - t.Unifs[i].Min
		}
		integral := sizeSpace * p.alpha[0]
		for i := 0; i < p.order; i++ {
			for j := 0; j < dim; j++ {
				max := t.Unifs[j].Max
				min := t.Unifs[j].Min
				pow := float64(i + 2) // 1 for order offset and 1 from integral
				inc := p.alpha[1+i*dim+j] * sizeSpace / (max - min) *
					(1.0 / pow) * (math.Pow(max, pow) - math.Pow(min, pow))
				integral += inc
			}
		}
		return integral / sizeSpace
	case distribution.IndependentGaussian:
		dim := t.Dim()
		if p.order > 3 {
			panic("Gaussian not programmed with order > 3")
		}
		// The expected value from the constant term
		ev := p.alpha[0]

		for i := 0; i < p.order; i++ {
			for j := 0; j < dim; j++ {
				a := p.alpha[1+i*dim+j]
				m := t.Norms[j].Mu
				s := t.Norms[j].Sigma
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

// puts in  1, x_1, x_2, ... x_n , x_1^2, ..., x_n^2, ... , x_1^order, ..., x_n^order
func coeffs(coeffvec []float64, x []float64, order int) {
	//nTerms := 1 + order*len(x)
	dim := len(x)
	coeffvec[0] = 1
	for i := 0; i < order; i++ {
		for j, v := range x {
			coeffvec[1+j+dim*i] = math.Pow(v, float64(i)+1)
		}
	}
}
