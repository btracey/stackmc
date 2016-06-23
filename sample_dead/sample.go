// package sample implements advanced sampling routines. Will be replaced by
// gonum/samplemv when that exists.
package sample

import (
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distmv"
)

type Rander interface {
	Rand([]float64) []float64
	Dim() int
}

type IID struct {
	Rander Rander
}

func (iid IID) Dim() int {
	return iid.Rander.Dim()
}

func (iid IID) RandBatch(x *mat64.Dense) {
	r, c := x.Dims()
	if c != iid.Dim() {
		panic("iid: bad dim")
	}
	for i := 0; i < r; i++ {
		iid.Rander.Rand(x.RawRowView(i))
	}
}

type Dimer interface {
	Dim() int
}

type MHProposal interface {
	ConditionalLogProb(x, y []float64) float64
	ConditionalRand(x, y []float64)
	Dimer
}

type LogProber interface {
	LogProb(x []float64) float64
	Dimer
}

type MetropolisHastings struct {
	target   LogProber
	proposal MHProposal
	initial  []float64
}

func NewMetropolisHastings(initial []float64, target LogProber, proposal MHProposal) MetropolisHastings {
	dim := len(initial)
	dimT := target.Dim()
	dimP := target.Dim()
	if dim != dimT {
		panic("dim mismatch")
	}
	if dimP != dimT {
		panic("dim mismatch")
	}
	c := make([]float64, len(initial))
	copy(c, initial)
	return MetropolisHastings{
		target:   target,
		proposal: proposal,
		initial:  c,
	}
}

func (mh MetropolisHastings) Dim() int {
	return mh.target.Dim()
}

func (mh MetropolisHastings) RandBatch(data *mat64.Dense) {
	current := make([]float64, len(mh.initial))
	copy(current, mh.initial)
	r, c := data.Dims()
	if c != len(mh.initial) {
		panic("dim mismatch")
	}
	target := mh.target
	proposal := mh.proposal

	proposed := make([]float64, len(current))

	currentLogProb := target.LogProb(current)
	for i := 0; i < r; i++ {
		proposal.ConditionalRand(proposed, current)
		proposedLogProb := target.LogProb(proposed)
		probTo := proposal.ConditionalLogProb(proposed, current)
		probBack := proposal.ConditionalLogProb(current, proposed)

		accept := math.Exp(proposedLogProb + probBack - probTo - currentLogProb)
		if accept > rand.Float64() {
			current, proposed = proposed, current
			currentLogProb = proposedLogProb
		}
		data.SetRow(i, current)
	}
}

type GaussianMH struct {
	dim int

	// Proposal distribution is a Gaussian ball around the current point.
	scale float64

	sigma *mat64.SymDense
}

func NewGaussianMH(scale float64, dim int) *GaussianMH {
	g := &GaussianMH{}
	g.dim = dim
	g.scale = scale

	g.sigma = mat64.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		g.sigma.SetSym(i, i, scale)
	}
	return g
}

func (g *GaussianMH) Dim() int {
	return g.dim
}

func (g *GaussianMH) ConditionalRand(x, y []float64) {
	n, ok := distmv.NewNormal(y, g.sigma, nil)
	if !ok {
		panic("bad sigma")
	}
	n.Rand(x)
}

func (g *GaussianMH) ConditionalLogProb(x, y []float64) float64 {
	n, ok := distmv.NewNormal(y, g.sigma, nil)
	if !ok {
		panic("bad sigma")
	}
	return n.LogProb(x)
}
