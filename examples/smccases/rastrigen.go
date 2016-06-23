package smccases

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/samplemv"
)

type MCMCTarget struct {
	Temp float64
	Func func(x []float64) float64
}

func GetSampler(initial []float64, m MCMCTarget, size float64, burnin int, rate int) samplemv.MetropolisHastingser {
	dim := len(initial)
	sigma := mat64.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		sigma.SetSym(i, i, size)
	}
	prop, ok := samplemv.NewProposalNormal(sigma, nil)
	if !ok {
		panic("bad sigma")
	}
	mher := samplemv.MetropolisHastingser{
		Initial:  initial,
		Target:   m,
		Proposal: prop,
		Src:      nil,

		BurnIn: burnin,
		Rate:   rate,
	}
	return mher
}

func SamplerEV(mh samplemv.MetropolisHastingser, f func(x []float64) float64, dim, samples int) float64 {
	batch := mat64.NewDense(samples, dim, nil)
	mh.Sample(batch)
	var ev float64
	for i := 0; i < samples; i++ {
		ev += f(batch.RawRowView(i))
	}
	ev /= float64(samples)
	return ev
}

func (m MCMCTarget) LogProb(x []float64) float64 {
	f := m.Func(x)
	return -f / m.Temp
}

type Rastrigin struct{}

func (Rastrigin) Func(x []float64) float64 {
	A := 10.0
	f := A * float64(len(x))
	for _, v := range x {
		f += v*v - A*math.Cos(2*math.Pi*v)
	}
	return f
}
