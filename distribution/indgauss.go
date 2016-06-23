package distribution

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distuv"
)

type ScoreInputer interface {
	ScoreInput(deriv, x []float64) []float64
}

// TODO(btracey): Replace these with samplemv.IIDer when it exists.

// IndependentGaussian is a Gaussian distribution where the
// dimenisons are independent from one another.
type IndependentGaussian struct {
	Norms []distuv.Normal
}

func (ind IndependentGaussian) Rand(x []float64) []float64 {
	if x == nil {
		x = make([]float64, len(ind.Norms))
	}
	if len(x) != len(ind.Norms) {
		panic("fit: length mismatch")
	}
	for i := range x {
		x[i] = ind.Norms[i].Rand()
	}
	return x
}

func (i IndependentGaussian) Dim() int {
	return len(i.Norms)
}

func (ind IndependentGaussian) LogProb(x []float64) float64 {
	if len(x) != len(ind.Norms) {
		panic("fit: length mismatch")
	}
	var logprob float64
	for i, v := range x {
		logprob += ind.Norms[i].LogProb(v)
	}
	return logprob
}

func (ind IndependentGaussian) Prob(x []float64) float64 {
	return math.Exp(ind.LogProb(x))
}

func (ind IndependentGaussian) Sample(data *mat64.Dense) {
	nSamples, _ := data.Dims()
	for i := 0; i < nSamples; i++ {
		ind.Rand(data.RawRowView(i))
	}
}

func (ind IndependentGaussian) Quantile(x []float64, p []float64) []float64 {
	if x == nil {
		x = make([]float64, len(p))
	}
	if len(x) != len(p) {
		panic("len mismatch")
	}
	for i, v := range p {
		x[i] = ind.Norms[i].Quantile(v)
	}
	return x
}

func (ind IndependentGaussian) ScoreInput(deriv, x []float64) []float64 {
	if deriv == nil {
		deriv = make([]float64, ind.Dim())
	}
	if len(deriv) != ind.Dim() {
		panic("len mismatch")
	}
	if len(x) != ind.Dim() {
		panic("len mismatch")
	}
	for i, xi := range x {
		deriv[i] = ind.Norms[i].ScoreInput(xi)
	}
	return deriv
}

type Uniform struct {
	Unifs []distuv.Uniform
}

func (u Uniform) Dim() int {
	return len(u.Unifs)
}

func (u Uniform) Rand(x []float64) []float64 {
	if x == nil {
		x = make([]float64, len(u.Unifs))
	}
	if len(x) != len(u.Unifs) {
		panic("fit: length mismatch")
	}
	for i := range x {
		x[i] = u.Unifs[i].Rand()
	}
	return x
}

func (u Uniform) Sample(data *mat64.Dense) {
	nSamples, _ := data.Dims()
	for i := 0; i < nSamples; i++ {
		u.Rand(data.RawRowView(i))
	}
}

func (u Uniform) LogProb(x []float64) float64 {
	if len(x) != len(u.Unifs) {
		panic("fit: length mismatch")
	}
	var logprob float64
	for i, v := range x {
		logprob += u.Unifs[i].LogProb(v)
	}
	return logprob
}

func (u Uniform) Prob(x []float64) float64 {
	return math.Exp(u.LogProb(x))
}

func (u Uniform) Quantile(x []float64, p []float64) []float64 {
	if x == nil {
		x = make([]float64, len(p))
	}
	if len(x) != len(p) {
		panic("len mismatch")
	}
	for i, v := range p {
		x[i] = u.Unifs[i].Quantile(v)
	}
	return x
}
