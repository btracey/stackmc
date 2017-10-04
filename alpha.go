package stackmc

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

// AlphaComputer uses the predictions to compute α for each fold. The returned
// alpha is indexed first by fold, and then by fit within that fold.
//
// The computed alpha should respect the information in Folds, using only the data
// specified by the Assess field of the fold.
type AlphaComputer interface {
	ComputeAlpha(xs mat.Matrix, fs, weights []float64, p distmv.RandLogProber, folds []Fold, fps []FoldPrediction, evs [][]float64) [][]float64
}

// SingleAlpha asigns a single alpha to all of the folds of each fit. It concatenates
// the predictions for the Assess indices across all of the folds, and calculates
// alpha as
//  α = A^-1 b
//  A_ij = cov(g_i, g_j)
//  b_i  = cov(f,g_i)
// the optimal estimator for a fixed control variate.
type SingleAlpha struct{}

var _ AlphaComputer = SingleAlpha{}

func (sa SingleAlpha) ComputeAlpha(xs mat.Matrix, f, weights []float64, d distmv.RandLogProber, folds []Fold, fps []FoldPrediction, evs [][]float64) [][]float64 {
	nFitters := len(evs[0])
	var totalPoints int
	for i := range folds {
		totalPoints += len(folds[i].Assess)
	}
	assessData := mat.NewDense(totalPoints, nFitters+1, nil)
	var weightData []float64
	if weights != nil {
		weightData = make([]float64, totalPoints)
	}
	var count int
	for i := range folds {
		for _, v := range folds[i].Assess {
			// First column is the actual function value
			assessData.Set(count, 0, f[v])
			if weights != nil {
				weightData[count] = weights[v]
			}

			// Rest of the columns are the fitter prections, possibly adjusted
			// by the fit expected value.
			idx, ok := fps[i].ToUniqueIdx[v]
			if !ok {
				panic("index missing in unique")
			}
			for k := 0; k < nFitters; k++ {
				pred := fps[i].Predictions[k][idx]
				assessData.Set(count, k+1, pred)
			}
			count++
		}
	}
	if count != totalPoints {
		panic("bad count")
	}

	covmat := stat.CovarianceMatrix(nil, assessData, weightData)

	// The optimal alpha is A\b where b is variance of f with the fitter (row/column 0)
	// and A are the covariances among the fitters (1:end, 1:end).
	covarWithF := mat.NewVecDense(nFitters, nil)
	for i := 0; i < nFitters; i++ {
		covarWithF.SetVec(i, covmat.At(0, i+1))
	}
	fitterCovar := covmat.SliceSquare(1, nFitters+1).(mat.Symmetric)
	var chol mat.Cholesky
	ok := chol.Factorize(fitterCovar)
	if !ok {
		// TODO(btracey): Handle error.
		panic("stackmc: cholesky error")
	}

	alpha := make([]float64, nFitters)
	alphaVec := mat.NewVecDense(len(alpha), alpha)

	err := chol.SolveVec(alphaVec, covarWithF)
	if err != nil {
		panic("stackmc: cholesky error")
	}

	// Set the same alpha for each fit across all of the folds.
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = make([]float64, nFitters)
		copy(alphas[i], alpha)
	}
	return alphas
}
