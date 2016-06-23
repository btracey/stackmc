package fit

import (
	"github.com/btracey/stackmc"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distmv"
)

// Gaussian fits the data with a multivariate Gaussian.
type Gaussian struct {
}

func (g Gaussian) FitDist(x mat64.Matrix, f, p []float64, inds []int) stackmc.DistPredictor {
	_, dim := x.Dims()
	xin := mat64.NewDense(len(inds), dim, nil)
	for i, idx := range inds {
		xin.SetRow(i, mat64.Row(nil, idx, x))
	}

	mean := make([]float64, dim)
	for i := range mean {
		col := mat64.Col(nil, i, xin)
		mean[i] = stat.Mean(col, nil)
	}

	cov := stat.CovarianceMatrix(nil, xin, nil)

	normal, ok := distmv.NewNormal(mean, cov, nil)
	if !ok {
		panic("bad fit")
	}
	return GaussianPred{normal}
}

type GaussianPred struct {
	Normal *distmv.Normal
}

func (g GaussianPred) Prob(x []float64) float64 {
	return g.Normal.Prob(x)
}

/*
func (g GaussianPred) Integrable(fun stackmc.Function) bool {
	return false
}

func (g GaussianPred) ExpectedValue(fun stackmc.Function) float64 {
	panic("shouldn't be here")
	return math.NaN()
}
*/

func (g GaussianPred) Sample(x *mat64.Dense) {
	nSamples, _ := x.Dims()
	for i := 0; i < nSamples; i++ {
		samp := x.RawRowView(i)
		g.Normal.Rand(samp)
	}
}
