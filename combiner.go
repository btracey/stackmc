package stackmc

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/gonum/stat/samplemv"
)

// Combiner combines the folds together to produce an expected value and an error
// estimate.
type Combiner interface {
	Combine(evAll []float64, foldEVs, alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) (ev, stdErr float64)
}

// BasicCombiner uses the average of the folds, the held-out Monte Carlo average
// and those held-out numbers as error bars.
type BasicCombiner struct{}

func (b BasicCombiner) Combine(evAll []float64, foldEVs, alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) (ev, stdErr float64) {
	foldCombinedEvs := basicFoldEVs(evAll, foldEVs, alpha, settings)
	avgEv := stat.Mean(foldCombinedEvs, nil)
	// Concatenate all the held-out differences.
	nFitters := len(alpha[0])
	fminus := make([]float64, 0)
	for i, fold := range folds {
		for _, idx := range fold.Update {
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			for l := 0; l < nFitters; l++ {
				truth -= alpha[i][l] * predictions[i][l][predIdx]
			}
			fminus = append(fminus, truth)
		}
	}
	ev = avgEv + stat.Mean(fminus, nil)
	stdErr = stat.StdDev(fminus, nil)
	stdErr = stat.StdErr(stdErr, float64(len(fminus)))
	return ev, stdErr
}

// CorrectorCombiner uses the average of the folds and an inner combiner
type CorrectorCombiner struct {
	Corrector Corrector
}

func (b CorrectorCombiner) Combine(evAll []float64, foldEVs, alpha [][]float64, uniqueMaps []map[int]int, predictions [][][]float64, d samplemv.Sampler, x mat64.Matrix, fitters []Fitter, f []float64, folds []Fold, settings *Settings) (ev, stdErr float64) {
	foldCombinedEVs := basicFoldEVs(evAll, foldEVs, alpha, settings)
	corrections := b.Corrector.Correct(alpha, uniqueMaps, predictions, d, x, fitters, f, folds, settings)
	for i, v := range corrections {
		foldCombinedEVs[i] += v
	}
	return stat.Mean(foldCombinedEVs, nil), math.NaN()
}

func basicFoldEVs(evAll []float64, evs, alpha [][]float64, settings *Settings) []float64 {
	nFolds := len(evs)
	nFitters := len(evAll)
	foldEVs := make([]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		var ev float64
		if settings.UpdateFull {
			// Use the EV from the fit to all
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evAll[j]
			}
		} else {
			// Use the EV from this particular fold
			for j := 0; j < nFitters; j++ {
				ev += alpha[i][j] * evs[i][j]
			}
		}
		foldEVs[i] = ev
	}
	return foldEVs
}
