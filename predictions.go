package stackmc

import (
	"math"
	"sort"
	"sync"

	"github.com/gonum/matrix/mat64"
)

// uniquePredMaps returns the unique indices that need to be predicted per fold
// (because the same indices could be present per training). The slice of slice
// returned goes from the unique index in that fold to the global data index. The
// map returned goes from the global index to the local unique index. Each is
// per-fold.
//
// allTrain is the set of the unique training indexes across all of the folds.
func findUniqueIndexs(folds []Fold) (uniques [][]int, uniqueMaps []map[int]int, allTrain []int) {
	nFolds := len(folds)
	// Mapping from global index to local unique index.
	uniqueMaps = make([]map[int]int, nFolds)
	// Mapping from unique index to global index
	uniques = make([][]int, nFolds)
	for i := range uniqueMaps {
		uniqueMaps[i] = make(map[int]int)
		uniques[i] = make([]int, 0, len(folds[i].Assess))
	}

	// The indices that need predictions are those in either Assess or Update.
	for i := 0; i < nFolds; i++ {
		for _, v := range folds[i].Assess {
			if _, ok := uniqueMaps[i][v]; ok {
				continue
			}
			uniqueMaps[i][v] = len(uniques[i])
			uniques[i] = append(uniques[i], v)
		}
		for _, v := range folds[i].Update {
			if _, ok := uniqueMaps[i][v]; ok {
				continue
			}
			uniqueMaps[i][v] = len(uniques[i])
			uniques[i] = append(uniques[i], v)
		}
	}

	m := make(map[int]struct{})
	for i := 0; i < nFolds; i++ {
		for _, v := range folds[i].Train {
			m[v] = struct{}{}
		}
	}
	allTrain = make([]int, len(m))
	var count int
	for idx := range m {
		allTrain[count] = idx
		count++
	}
	sort.Ints(allTrain)
	return uniques, uniqueMaps, allTrain
}

// trainAndPredict creates the fitters for all and per fold, and creates predictons
// at all the needed points.
// Indexed by [fold][fitter] for two-element ones.
// Indexed by [fold][fitter][point] for the predictions.
func trainAndPredict(allTrain []int, uniques [][]int, x mat64.Matrix, f []float64,
	folds []Fold, fitters []Fitter, d Distribution, settings *Settings) (
	predAll []Predictor, evAll []float64, preds [][]Predictor, evs [][]float64, predictions [][][]float64) {
	nFolds := len(folds)
	nFit := len(fitters)
	_, dim := x.Dims()

	evmul := settings.EstimateFitEV
	nTrain := len(allTrain)

	// Train on all the indices.
	var wg sync.WaitGroup
	predAll = make([]Predictor, nFit)
	evAll = make([]float64, nFit)
	for i := range predAll {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			pred := fitters[i].Fit(x, f, d, allTrain)
			predAll[i] = pred
			evAll[i] = predictorExpectedValue(pred, d, evmul, nTrain, dim)
		}(i)
	}

	preds = make([][]Predictor, nFolds)
	predictions = make([][][]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		preds[i] = make([]Predictor, nFit)
		predictions[i] = make([][]float64, nFit)
		for j := 0; j < nFit; j++ {
			predictions[i][j] = make([]float64, len(uniques[i]))
			for k := range predictions[i][j] {
				predictions[i][j][k] = math.NaN()
			}
		}
	}

	evs = make([][]float64, nFolds)
	// Train each fitter individually and make the individual predictions.
	for i := range folds {
		evs[i] = make([]float64, nFit)
		for j := range fitters {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				fold := folds[i]
				fitter := fitters[j]

				pred := fitter.Fit(x, f, d, fold.Train)
				preds[i][j] = pred
				row := make([]float64, dim)
				for k, v := range uniques[i] {
					mat64.Row(row, v, x)
					predictions[i][j][k] = pred.Predict(row)
				}
				evs[i][j] = predictorExpectedValue(pred, d, evmul, nTrain, dim)
			}(i, j)
		}
	}
	wg.Wait()
	return
}

func predictorExpectedValue(pred Predictor, d Distribution, mul float64, number, dim int) float64 {
	if pred.Integrable(d) {
		return pred.ExpectedValue(d)
	}
	if mul == -1 {
		panic("stackmc: distribution not integrable")
	}
	s := int(float64(number) * mul)
	if s <= 0 {
		panic("bad number of ev samples")
	}
	samples := mat64.NewDense(s, dim, nil)
	d.Sample(samples)
	var ev float64
	for i := 0; i < s; i++ {
		ev += pred.Predict(samples.RawRowView(i))
	}
	ev /= float64(s)
	return ev
}
