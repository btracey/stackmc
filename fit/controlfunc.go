package fit

import (
	"fmt"
	"sync"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/distribution"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/samplemv"
)

type CFKernelOneD interface {
	Distance(x, y float64) float64
	Deriv(x, y float64) float64 // dk/dx at {x,y}
	Hessian(x, y float64) float64
}

type CFKernelOneDWrapper struct {
	CFKernelOneD
}

func (c CFKernelOneDWrapper) Distance(x, y []float64) float64 {
	if len(x) != 1 {
		panic("bad size")
	}
	return c.CFKernelOneD.Distance(x[0], y[0])
}

func (c CFKernelOneDWrapper) Deriv(deriv, x, y []float64) []float64 {
	if len(x) != 1 {
		panic("bad size")
	}
	if deriv == nil {
		deriv = make([]float64, 1)
	}
	deriv[0] = c.CFKernelOneD.Deriv(x[0], y[0])
	return deriv
}

func (c CFKernelOneDWrapper) Laplacian(x, y []float64) float64 {
	if len(x) != 1 {
		panic("bad size")
	}
	return c.CFKernelOneD.Hessian(x[0], y[0])
}

// ControlFunc implements the estimator based on
//  Control Functionals for Monte Carlo Integration (Oates, Girolami, Chopin)
type ControlFunc struct {
	Kernel CFKernel
	Noise  float64
}

type CFKernel interface {
	Distance(x, y []float64) float64
	Deriv(deriv, x, y []float64) []float64 // dk/dx at {x,y}
	Laplacian(x, y []float64) float64      // d^2 k / dx dy
}

func (cont ControlFunc) Fit(x mat64.Matrix, f []float64, d samplemv.Sampler, inds []int) stackmc.Predictor {
	_, c := x.Dims()
	subf := make([]float64, len(inds))
	for i, v := range inds {
		subf[i] = f[v]
	}
	subdata := mat64.NewDense(len(inds), c, nil)
	for i, v := range inds {
		subdata.SetRow(i, mat64.Row(nil, v, x))
	}
	return &ControlFuncPredictor{
		data:   subdata,
		f:      subf,
		kernel: cont.Kernel,
		noise:  cont.Noise,
		dist:   d,
	}
}

type ControlFuncPredictor struct {
	data   *mat64.Dense
	f      []float64
	kernel CFKernel
	//kmd    CFKernel
	noise float64
	dist  samplemv.Sampler

	once      sync.Once
	k0        *mat64.SymDense
	cholk0    *mat64.Cholesky
	cholk0f0  *mat64.Vector
	cholk0one *mat64.Vector
	ev        float64
}

/*
func (c *ControlFuncPredictor) Predict(x []float64) float64 {
	distsi := c.dist.(distribution.ScoreInputer)
	c.setk0calcev(distsi)

	// Set k1
	k1 := mat64.NewVector(len(c.f), nil)
	for i := 0; i < len(c.f); i++ {
		xi := c.data.At(i, 0)
		v := c.calculateKernel(xi, x[0], distsi)
		k1.SetVec(i, v)
	}

	// fhat = k_1 * (k0)^-1 f_0 + (1 - k1 * (k0)^-1 1)(ev)
	first := mat64.Dot(k1, c.cholk0f0)
	second := mat64.Dot(k1, c.cholk0one)
	pred := first + (1-second)*c.ev
	return pred
}
*/

func (c *ControlFuncPredictor) Predict(x []float64) float64 {
	distsi := c.dist.(distribution.ScoreInputer)
	c.setk0calcev(distsi)

	// Set k1
	k1 := mat64.NewVector(len(c.f), nil)
	for i := 0; i < len(c.f); i++ {
		xi := c.data.RawRowView(i)
		v := c.calculateKernel(xi, x, distsi)
		k1.SetVec(i, v)
	}

	// fhat = k_1 * (k0)^-1 f_0 + (1 - k1 * (k0)^-1 1)(ev)
	first := mat64.Dot(k1, c.cholk0f0)
	second := mat64.Dot(k1, c.cholk0one)
	pred := first + (1-second)*c.ev
	return pred
}

func (c *ControlFuncPredictor) Integrable(dist samplemv.Sampler) bool {
	_ = dist.(distribution.ScoreInputer)
	return true
}

func (c *ControlFuncPredictor) ExpectedValue(dist samplemv.Sampler) float64 {
	// Control Functionals for Monte Carlo Integration by Oates, Girolami and Chopin.
	distsi, ok := dist.(distribution.ScoreInputer)
	if !ok {
		panic("bad score inputter")
	}
	c.setk0calcev(distsi)
	// done this way because Predict also needs these values
	return c.ev
}

func (c *ControlFuncPredictor) setk0calcev(distsi distribution.ScoreInputer) {
	c.once.Do(func() {
		// Section 2.3.1
		//  k_0(x,x') = d^2 / dxdx' k(x,x') + u(x) dk/dx' + u(x') dk/dx + u(x)u(x')k(x,x')
		m, _ := c.data.Dims()
		ko := mat64.NewSymDense(m, nil)
		for i := 0; i < m; i++ {
			for j := i; j < m; j++ {
				//xi := c.data.At(i, 0)
				//xj := c.data.At(j, 0)
				xi := c.data.RawRowView(i)
				xj := c.data.RawRowView(j)
				v := c.calculateKernel(xi, xj, distsi)
				if i == j {
					v += c.noise
				}
				ko.SetSym(i, j, v)
			}
		}
		c.k0 = ko

		// Set EV
		// Numerator: 1^T (K_0)^-1 * f_0
		// Denominator: 1 + 1^T (K_0)^-1 * 1^T
		var chol mat64.Cholesky
		ok := chol.Factorize(c.k0)
		if !ok {
			fmt.Println(c.data)
			fmt.Printf("%0.4v\n", mat64.Formatted(c.k0))
			fmt.Println(mat64.Cond(c.k0, 1))
			panic("bad factorization")
		}
		fvec := mat64.NewVector(len(c.f), c.f)
		var ans mat64.Vector
		err := ans.SolveCholeskyVec(&chol, fvec)
		if err != nil {
			panic("bad solve")
		}
		num := mat64.Sum(&ans)

		o := make([]float64, len(c.f))
		for i := range o {
			o[i] = 1
		}
		ones := mat64.NewVector(len(c.f), o)
		var ans2 mat64.Vector
		err = ans2.SolveCholeskyVec(&chol, ones)
		if err != nil {
			panic("bad solve")
		}
		den := 1 + mat64.Sum(&ans2)
		ev := num / den
		c.ev = ev
		c.cholk0 = &chol
		c.cholk0f0 = &ans
		c.cholk0one = &ans2
	})
}

/*
func (c *ControlFuncPredictor) calculateKernel(xi, xj float64, dist distribution.ScoreInputer) float64 {
	k := c.kernel.Distance(xi, xj)
	kdxi := c.kernel.Deriv(xi, xj)
	kdxj := c.kernel.Deriv(xj, xi)
	kh := c.kernel.Hessian(xi, xj)

	uxi := dist.ScoreInput(nil, []float64{xi})[0]
	uxj := dist.ScoreInput(nil, []float64{xj})[0]
	//fmt.Println(k, kdxi, kdxj, kh, uxi, uxj)

	return kh + uxi*kdxj + uxj*kdxi + uxi*uxj*k
}
*/

func (c *ControlFuncPredictor) calculateKernel(xi, xj []float64, dist distribution.ScoreInputer) float64 {
	//  k_0(x,x') = ∇^2 k(x,x') + u(x) dk/dx' + u(x') dk/dx + u(x)u(x')k(x,x')
	k := c.kernel.Distance(xi, xj)
	kdxi := c.kernel.Deriv(nil, xi, xj)
	kdxj := c.kernel.Deriv(nil, xj, xi)
	kh := c.kernel.Laplacian(xi, xj)

	uxi := dist.ScoreInput(nil, xi)
	uxj := dist.ScoreInput(nil, xj)

	return kh + floats.Dot(uxi, kdxj) + floats.Dot(uxj, kdxi) + floats.Dot(uxi, uxj)*k
}
