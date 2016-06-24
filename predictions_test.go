package stackmc

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distmv"
	"github.com/gonum/stat/samplemv"
)

type meanFitter struct {
	Intergable bool
}

func (m meanFitter) Fit(x mat64.Matrix, f []float64, d Distribution, inds []int) Predictor {
	var ev float64
	for _, v := range inds {
		ev += f[v]
	}
	ev /= float64(len(inds))
	if m.Intergable {
		return meanPredInt{ev}
	}
	return meanPredNonInt{ev}
}

type meanPredInt struct {
	Mean float64
}

func (m meanPredInt) Predict(x []float64, d Distribution) float64 {
	// Here for testing purposes.
	return m.Mean - (x[0] - 0.5)
}

func (m meanPredInt) Integrable(d Distribution) bool {
	return true
}

func (m meanPredInt) ExpectedValue(d Distribution) float64 {
	return m.Mean
}

type meanPredNonInt struct {
	Mean float64
}

func (m meanPredNonInt) Predict(x []float64, d Distribution) float64 {
	// Here for testing purposes.
	return m.Mean + (x[0] - 0.5)
}

func (m meanPredNonInt) Integrable(d Distribution) bool {
	return false
}

func (m meanPredNonInt) ExpectedValue(d Distribution) float64 {
	panic("shouldn't be here")
}

func TestPredictions(t *testing.T) {
	nSamp := 100
	dim := 3
	x := mat64.NewDense(nSamp, dim, nil)
	f := make([]float64, nSamp)
	for i := 0; i < nSamp; i++ {
		var sum float64
		for j := 0; j < dim; j++ {
			v := rand.Float64()
			x.Set(i, j, v)
			sum += v
		}
		f[i] = sum / float64(dim)
	}
	nFolds := 5
	folds := make([]Fold, nFolds)
	nTrain := 70
	nTest := nSamp - nTrain
	for i := 0; i < nFolds; i++ {
		folds[i].Train = make([]int, nTrain)
		folds[i].Update = make([]int, nTest)
		folds[i].Assess = make([]int, nTest)
		perm := rand.Perm(nSamp)
		for j := 0; j < nTrain; j++ {
			folds[i].Train[j] = perm[j]
		}
		for j := 0; j < nTest; j++ {
			idx := nTrain + j
			folds[i].Update[j] = perm[idx]
			folds[i].Assess[j] = perm[idx]
		}
	}

	// Get the unique indices like one would, and run.
	bnds := make([]distmv.Bound, dim)
	for i := range bnds {
		bnds[i].Min = 0
		bnds[i].Max = 1
	}
	d := samplemv.IIDer{
		distmv.NewUniform(bnds, nil),
	}
	settings := &Settings{}
	settings.EstimateFitEV = 10
	uniques, uniqueMaps, allTrain := findUniqueIndexs(folds)
	_ = uniqueMaps
	predAll, evAll, preds, evs, predictions := trainAndPredict(allTrain, uniques, x, f, folds, []Fitter{meanFitter{true}, meanFitter{false}}, d, settings)
	_, _, _, _, _ = predAll, evAll, preds, evs, predictions
	tol := 5e-2
	ans := 0.5

	for _, v := range predAll {
		if v == nil {
			t.Errorf("nil pred all")
		}
	}

	// Check that the evs are close to the true ev.
	for i, v := range evAll {
		if math.Abs(v-ans) > tol {
			t.Errorf("EV mismatch pred %v, got %v, want %v", i, v, ans)
		}
	}

	for _, fold := range preds {
		for _, v := range fold {
			if v == nil {
				t.Errorf("nil predictor")
			}
		}
	}

	// Test that the different folds have different predictions.
	m := make(map[float64]struct{})
	for _, fold := range evs {
		for _, ev := range fold {
			_, ok := m[ev]
			if ok {
				t.Errorf("Two fold EVs the same")
			}
			m[ev] = struct{}{}
			if math.Abs(ev-ans) > tol {
				t.Errorf("fold EV mismatch")
			}
		}
	}

	// Check that the predictions are correct.
	for i, fold := range folds {
		for _, idx := range fold.Update {
			ui := uniqueMaps[i][idx]
			ev0 := evs[i][0]
			pred0 := predictions[i][0][ui]
			want0 := ev0 - x.At(idx, 0) + 0.5
			if math.Abs(want0-pred0) > 1e-14 {
				t.Errorf("prediction mismatch. Fit %v, want %v, got %v", 0, want0, pred0)
			}
			pred1 := predictions[i][1][ui]
			// Need to use ev0 because ev1 is computed from independent random samples.
			want1 := ev0 + x.At(idx, 0) - 0.5
			if math.Abs(want1-pred1) > 1e-14 {
				t.Errorf("prediction mismatch. Fit %v, want %v, got %v", 1, want1, pred1)
			}
		}
	}
}

func TestFindUniqueIndexes(t *testing.T) {
	nTest := 10
	sz := 30
	insz := 50
	for test := 0; test < nTest; test++ {
		nFolds := rand.Intn(5) + 1
		folds := make([]Fold, nFolds)
		for i := range folds {
			folds[i].Train = make([]int, rand.Intn(sz))
			randomIndices(folds[i].Train, insz)
			folds[i].Assess = make([]int, rand.Intn(sz))
			randomIndices(folds[i].Assess, insz)
			folds[i].Update = make([]int, rand.Intn(sz))
			randomIndices(folds[i].Update, insz)
		}
		uniques, uniqueMaps, _ := findUniqueIndexs(folds)

		for i := range uniques {
			// Check that the indices are all unique.
			if !allUniqueInt(uniques[i]) {
				t.Errorf("index overlap uniques")
			}

			if len(uniqueMaps[i]) != len(uniques[i]) {
				t.Errorf("different number of unique indices")
			}
			for j, v := range uniques[i] {
				if uniqueMaps[i][v] != j {
					t.Errorf("Global <--> Local correspondence mismatch", j, v)
				}
			}
			lists := [][]int{folds[i].Assess, folds[i].Update}
			if !verifyMap(lists, uniqueMaps[i]) {
				t.Errorf("bad map")
			}
		}
	}
}

// verifyMap guarantees that all of the indices are in the map and that there
// are no extra elements
func verifyMap(lists [][]int, m map[int]int) bool {
	in := make([]bool, len(m))
	for _, list := range lists {
		for _, v := range list {
			// Confirm the element is in the map.
			if _, ok := m[v]; !ok {
				return false
			}
			// Use to mark that the element is there.
			in[m[v]] = true
		}
	}
	for _, b := range in {
		if !b {
			return false
		}
	}
	return true
}

func allUniqueInt(s []int) bool {
	m := make(map[int]struct{})
	for _, v := range s {
		_, ok := m[v]
		if ok {
			return false
		}
		m[v] = struct{}{}
	}
	return true
}

func randomIndices(s []int, sz int) {
	for i := range s {
		s[i] = rand.Intn(sz)
	}
}
