package stackmc

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/samplemv"
)

type AverageHeldOut struct{}

func (a AverageHeldOut) Correct(alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) []float64 {
	nFitters := len(alpha[0])
	corrections := make([]float64, len(folds))
	for i := range folds {
		var ev float64
		z := 1 / float64(len(folds[i].Update))
		for _, idx := range folds[i].Update {
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			for l := 0; l < nFitters; l++ {
				pred := predictions[i][l][predIdx]
				// This seems wrong, truth added multiple times.
				ev += z * (truth - alpha[i][l]*pred)
			}
		}
		corrections[i] = ev
	}
	return corrections
}

// Makes a fit to the f - alpha * g using the value at the held-out data samples.
type FitInner struct{}

func (FitInner) Correct(alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) []float64 {
	nFitters := len(alpha[0])
	if nFitters != 1 {
		panic("only coded for one fitter")
	}
	newfs := make([]float64, len(f))
	for i := range newfs {
		newfs[i] = math.NaN()
	}
	indsNewF := make([]int, 0)
	// Construct the new data points as f - alpha*g
	for i, fold := range folds {
		for _, idx := range fold.Update {
			if !math.IsNaN(newfs[idx]) {
				panic("only coded when no overlap in held-out predictions")
			}
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			for l := 0; l < nFitters; l++ {
				pred := predictions[i][l][predIdx]
				truth -= alpha[i][l] * pred
			}
			newfs[idx] = truth
			indsNewF = append(indsNewF, idx)
		}
	}
	ev := FitExpectedValue(fitters[0], d, x, newfs, indsNewF, settings.EstimateFitEV)
	correction := make([]float64, len(folds))
	for i := range correction {
		correction[i] = ev
	}
	return correction
}

// Does a fit to the error in each fold individually
type FitInnerEach struct{}

func (fi FitInnerEach) Correct(alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) []float64 {
	_, dim := x.Dims()
	nFitters := len(alpha[0])
	if nFitters != 1 {
		panic("only coded for one fitter")
	}
	correction := make([]float64, len(folds))
	for i, fold := range folds {
		ndata := len(fold.Update)
		newfs := make([]float64, ndata)
		idxs := make([]int, ndata)
		newxs := mat64.NewDense(ndata, dim, nil)
		for j, idx := range fold.Update {
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			for l := 0; l < nFitters; l++ {
				pred := predictions[i][l][predIdx]
				truth -= alpha[i][l] * pred
			}
			newfs[j] = truth
			idxs[j] = j
			newxs.SetRow(j, mat64.Row(nil, idx, x))
		}
		ev := FitExpectedValue(fitters[0], d, newxs, newfs, idxs, settings.EstimateFitEV)
		correction[i] = ev
	}
	return correction
}

// Run StackMC on the inner loop using f - alpha g. This is weird when alpha
// is non-uniform and the same point can be held-out mulitple times.
type StackMCRecursive struct{}

func (s StackMCRecursive) Correct(alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) []float64 {
	nFitters := len(alpha[0])

	newfs := make([]float64, len(f))
	for i := range newfs {
		newfs[i] = math.NaN()
	}

	// Construct the new data points as f - alpha*g
	for i, fold := range folds {
		for _, idx := range fold.Update {
			if !math.IsNaN(newfs[idx]) {
				panic("only coded when no overlap in held-out predictions")
			}
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			for l := 0; l < nFitters; l++ {
				pred := predictions[i][l][predIdx]
				truth -= alpha[i][l] * pred
			}
			newfs[idx] = truth
		}
	}

	// Use StackMC to estimate this integral
	newSettings := &Settings{
		settings.UpdateFull,
		settings.Concurrent,
		settings.AlphaComputer,
		settings.EstimateFitEV,
		nil, // Don't recurse again
	}

	result := Estimate(d, x, newfs, fitters, folds, newSettings)

	correction := make([]float64, len(folds))
	for i := range correction {
		correction[i] = result.EV
	}
	return correction
}
