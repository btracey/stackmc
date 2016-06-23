package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/distribution"
	"github.com/btracey/stackmc/fit"
	"github.com/btracey/stackmc/fold"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distuv"
)

func main() {
	// Look at the average value of f - g for latin hypercube fold vs. the average
	// value of f - g for non-latin hypercube. Similarly, the average covariance
	// between ghat and g for the two versions.
	//rand.Seed(time.Now().UnixNano())
	rand.Seed(201605112)

	nFolds := 5
	nRuns := 100000

	nSamples := 500
	dim := 1

	fitter := &fit.Polynomial{1}
	function := func(x []float64) float64 {
		if len(x) != 1 {
			panic("bad size")
		}
		return (x[0] - 0.2) * (x[0] - 0.2)
	}

	sampler := getUniform(dim, 0, 1)
	/*
		sampler := samplemv.LatinHypercuber{
			Q: getUniform(dim, 0, 1),
		}
	*/

	polysamp := getUniform(dim, 0, 1)

	var allev []float64
	var allpreds []float64
	var allfs []float64
	var diffs []float64
	var evdiff []float64
	var evdiffdiff []float64
	for i := 0; i < nRuns; i++ {
		x := mat64.NewDense(nSamples, dim, nil)
		sampler.Sample(x)

		f := make([]float64, nSamples)
		for j := range f {
			f[j] = function(x.RawRowView(j))
		}

		folds := fold.KFold{nFolds}.Folds(nSamples)

		for j, fold := range folds {
			_ = j
			predictor := fitter.Fit(x, f, fold.Train)

			ev := predictor.ExpectedValue(polysamp)
			//fmt.Println("ev is ", ev)

			//preds := make([]float64, len(fold.Update))
			for _, idx := range fold.Update {
				pred := predictor.Predict(x.RawRowView(idx))
				//fmt.Println(pred, f[idx])
				allev = append(allev, ev)
				allpreds = append(allpreds, pred)
				allfs = append(allfs, f[idx])
				diffs = append(diffs, f[idx]-pred)
				evdiff = append(evdiff, pred-ev)
				evdiffdiff = append(evdiffdiff, f[idx]-(pred-ev))
				//fmt.Println("Fold", j, "pred-ev:", pred-ev)
			}
			//plotFold("fold_"+strconv.Itoa(j), predictor, x, f, fold, function)
		}
		//fmt.Printf("%#v\n", folds)
	}
	//fmt.Println(stat.Covariance(allfs, allpreds, nil) / stat.Variance(allpreds, nil))
	//fmt.Println(stat.Mean(diffs, nil))
	fmt.Println("Average pred-ev", stat.Mean(evdiff, nil)) // bias
	//fmt.Println(stat.Mean(evdiffdiff, nil)) // true expected + bias
	//fmt.Println(stat.Mean(allfs, nil))      // true expected value
	fmt.Println(stat.Correlation(allfs, evdiff, nil))
	//fmt.Println(stat.Covariance(allev, allpreds, nil) - stat.Variance(allev, nil))
}

func plotFold(name string, predictor stackmc.Predictor, x mat64.Matrix, f []float64, fold stackmc.Fold, function func(x []float64) float64) {
	_, dim := x.Dims()
	if dim != 1 {
		panic("only coded for one")
	}

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.X.Min = 0
	p.X.Max = 1
	p.Y.Min = -0.3
	p.Y.Max = 1

	truth := func(x float64) float64 {
		return function([]float64{x})
	}

	pred := func(x float64) float64 {
		return predictor.Predict([]float64{x})
	}

	tf := plotter.NewFunction(truth)
	pf := plotter.NewFunction(pred)
	p.Add(tf, pf)

	trainpts := make(plotter.XYs, len(fold.Train))
	for i, idx := range fold.Train {
		trainpts[i].X = mat64.Row(nil, idx, x)[0]
		trainpts[i].Y = f[idx]
	}

	testpts := make(plotter.XYs, len(fold.Update))
	for i, idx := range fold.Update {
		testpts[i].X = mat64.Row(nil, idx, x)[0]
		testpts[i].Y = f[idx]
	}

	rs, _ := plotter.NewScatter(trainpts)
	rs.Color = plotutil.SoftColors[0]

	rs.Shape = draw.CircleGlyph{}
	rs.Radius = 5

	es, _ := plotter.NewScatter(testpts)
	es.Color = plotutil.SoftColors[1]

	es.Shape = draw.PyramidGlyph{}
	es.Radius = 5

	p.Add(rs, es)

	p.Save(4*vg.Inch, 4*vg.Inch, name+".pdf")
}

func getUniform(dim int, min, max float64) distribution.Uniform {
	dist := distribution.Uniform{}
	for i := 0; i < dim; i++ {
		dist.Unifs = append(dist.Unifs, distuv.Uniform{Min: min, Max: max})
	}
	return dist
}
