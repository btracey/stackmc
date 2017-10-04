package stackmc

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Combiner combines the folds together to produce an expected value and an error estimate.
//
// evAll is the expected value for each fitter to all of the data.
// FoldEVs is the expected value for that particular fold and fitter (indexed)
// first by fold, and then by fitter. alphas is the same, except the data
// contains the alpha computed by that fold.
type Combiner interface {
	Combine(xs mat.Matrix, fs, weights []float64, p distmv.RandLogProber, folds []Fold, evAll []float64, foldEVs, alpha [][]float64, fps []FoldPrediction) (ev float64)
}

// BasicCombiner estimates an expected value for each fold by computing
//  α_k ĝ_k + 1/|assess_k| \sum_{x_i ∈ assess_k} f(x_i) - α_k g_k(x_i)
// for each fold, and then the final expected value as the average of
// the fold expected values.
type BasicCombiner struct{}

var _ Combiner = BasicCombiner{}

func (b BasicCombiner) Combine(xs mat.Matrix, f, weights []float64, p distmv.RandLogProber, folds []Fold, evAll []float64, foldEVs, alpha [][]float64, fps []FoldPrediction) (ev float64) {
	// Compute \sum_i \alpha_i ghat_i
	// If len(fold.Combine) = 0, that fold is ignored here.
	foldCombinedEvs := b.combineFitEVs(evAll, foldEVs, alpha)

	foldWeights := make([]float64, len(foldCombinedEvs))
	for i, fold := range folds {
		if len(fold.Correct) == 0 {
			foldWeights[i] = 0 // Don't use the folds that have nothing to correct.
		} else {
			foldWeights[i] = 1
		}
	}
	avgEv := stat.Mean(foldCombinedEvs, foldWeights)

	// Compute the correction term.
	//  \sum_i \sum_k w_k (f_k - alpha_i * g_k )
	// Where i is the sum is over all the folds, and k is the sum of all the
	// elements in Correct of that fold.
	nFitters := len(alpha[0])
	var fminus []float64
	var w []float64
	for i, fold := range folds {
		for _, idx := range fold.Correct {
			truth := f[idx]
			predIdx := fps[i].ToUniqueIdx[idx]
			for l := 0; l < nFitters; l++ {
				truth -= alpha[i][l] * fps[i].Predictions[l][predIdx]
			}
			fminus = append(fminus, truth)
			if weights != nil {
				w = append(w, weights[idx])
			}
		}
	}
	ev = avgEv + stat.Mean(fminus, w)
	return ev
	//stdErr = stat.StdDev(fminus, nil)
	//stdErr = stat.StdErr(stdErr, float64(len(fminus)))
	//return ev, stdErr
}

// combineFitEVs computes \alpha_i g_i for each fold.
func (b BasicCombiner) combineFitEVs(evAll []float64, evs, alpha [][]float64) []float64 {
	nFolds := len(evs)
	nFitters := len(evs[0])
	if alpha == nil {
		panic("alpha nil")
	}
	if len(alpha) != nFolds {
		panic("len(alpha) not equal to number of folds")
	}
	for i := range alpha {
		if len(alpha[i]) != nFitters {
			panic("alpha does not match nFitters")
		}
	}
	foldEVs := make([]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		var ev float64
		// Use the EV from this particular fold
		for j := 0; j < nFitters; j++ {
			ev += alpha[i][j] * evs[i][j]
		}
		foldEVs[i] = ev
	}
	return foldEVs
}
