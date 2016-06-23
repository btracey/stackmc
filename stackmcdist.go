package stackmc

import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

type DistFitter interface {
	FitDist(x mat64.Matrix, f, p []float64, inds []int) DistPredictor
}

type DistPredictor interface {
	// TODO(btracey): Should be log prob
	Prob(x []float64) float64
	//Integrable(fun Function) bool
	//ExpectedValue(fun Function) float64
	Sample(x *mat64.Dense)
}

type DistFunction interface {
	Func([]float64) float64
	Integrable(d DistPredictor) bool
	ExpectedValue(d DistPredictor) float64
}

func FitDistEV(fit DistFitter, fun DistFunction, x mat64.Matrix, f, p []float64, inds []int, evMult float64, evMin int) float64 {
	pred := fit.FitDist(x, f, p, inds)
	if fun.Integrable(pred) {
		return fun.ExpectedValue(pred)
	}
	if evMult == -1 {
		panic("stackmc: distribution not integrable")
	}
	_, dim := x.Dims()
	nSamples := len(inds)
	nNewsamp := int(float64(nSamples) * evMult)
	if nNewsamp < evMin {
		nNewsamp = evMin
	}
	xnew := mat64.NewDense(nNewsamp, dim, nil)
	pred.Sample(xnew)
	var ev float64
	for i := 0; i < nNewsamp; i++ {
		ev += fun.Func(xnew.RawRowView(i))
	}
	ev /= float64(nNewsamp)
	return ev
}

// EstimateDist estimates the expected value from a set of samples where the
// function is cheap and the probability distribution is expensive to evaluate.
func EstimateDist(fun DistFunction, x mat64.Matrix, f, p []float64, normalized bool, fitters DistFitter, folds []Fold) (ev float64) {
	nSamples, dim := x.Dims()
	if len(f) != nSamples {
		panic(errLen)
	}
	if len(p) != nSamples {
		panic(errLen)
	}

	nFolds := len(folds)
	if nFolds == 0 {
		panic("stackmc: no training folds")
	}

	// TODO(btracey): Allow more than one fitter.
	/*
		nFitters := len(fitters)
		if nFitters != 1 {
			// tried to code mostly to allow multiple fitters, but need to look at the
			// right way to combine multiple fitting algorithms.
			panic("estimate dist only coded for 1 fitter")
		}
	*/

	// Predict the probability value at all of the locations.
	// TODO(btracey): Make like the other code that is parallel and smart.
	predictors := make([]DistPredictor, nFolds)
	evs := make([]float64, nFolds)

	alphaPredictions := make([][]float64, nFolds)
	trueP := make([][]float64, nFolds)
	trueF := make([][]float64, nFolds)

	updatePredictions := make([][]float64, nFolds)
	row := make([]float64, dim)
	for i, fold := range folds {
		trueP[i] = make([]float64, len(fold.Assess))
		trueF[i] = make([]float64, len(fold.Assess))
		predictors[i] = fitters.FitDist(x, f, p, fold.Train)
		alphaPredictions[i] = make([]float64, len(fold.Assess))
		for j, idx := range fold.Assess {
			mat64.Row(row, idx, x)
			alphaPredictions[i][j] = predictors[i].Prob(row)
			trueP[i][j] = p[idx]
			trueF[i][j] = f[idx]
		}
		updatePredictions[i] = make([]float64, len(fold.Update))
		for j, idx := range fold.Update {
			mat64.Row(row, idx, x)
			updatePredictions[i][j] = predictors[i].Prob(row)
		}

		if !fun.Integrable(predictors[i]) {
			// Predict the outcome.
			//newsamp := nSamples * 100
			//if newsamp < 10000 {
			newsamp := 10000
			//}
			samples := mat64.NewDense(newsamp, dim, nil)
			predictors[i].Sample(samples)
			var ev float64
			for j := 0; j < newsamp; j++ {
				ev += fun.Func(samples.RawRowView(j))
			}
			ev /= float64(newsamp)
			evs[i] = ev
		} else {
			evs[i] = fun.ExpectedValue(predictors[i])
			//evs[i] = predictors[i].ExpectedValue(fun)
		}
	}

	// Compute Alpha
	alphas := computeAlphaDist(trueP, trueF, alphaPredictions)
	//fmt.Println("alphas", alphas)
	var zs []float64
	if normalized {
		zs = make([]float64, nFolds)
		for i := range zs {
			zs[i] = 1
		}
	} else {
		zs = estimateZ(trueP, alphaPredictions)
		//fmt.Println("z = ", zs)
	}

	// Estimate the EV.
	var evEst float64
	for i, fold := range folds {
		var corrEV float64
		for j, idx := range fold.Update {
			corrEV += f[idx] * (p[idx] - alphas[i]*updatePredictions[i][j]) / p[idx]
		}
		corrEV /= float64(len(fold.Update))

		evEst += alphas[i]*evs[i]*(1/zs[i]) + corrEV
	}
	evEst /= float64(len(folds))
	return evEst
}

// TODO(btracey): make these interfaces like the normal one so can easily exchange
// types.
// TODO(btracey):

// estimateZ estimates Z from the held out data.
// \int_p q/r p dx = 1/z
func estimateZ(p, preds [][]float64) (zs []float64) {
	var ps, qs []float64
	for i := range p {
		for j := range p[i] {
			ps = append(ps, p[i][j])
			qs = append(qs, preds[i][j])
		}
	}
	var z float64
	for i := range ps {
		z += qs[i] / ps[i]
	}
	z /= float64(len(ps))
	z = 1 / z
	zs = make([]float64, len(p))
	for i := range zs {
		zs[i] = z
	}
	return zs
}

func computeAlphaDist(p, f, preds [][]float64) (alphas []float64) {
	var fqps, fs []float64
	for i := range preds {
		for j := range preds[i] {
			fs = append(fs, f[i][j])
			fqp := f[i][j] * preds[i][j] / p[i][j]
			fqps = append(fqps, fqp)
		}
	}
	c := stat.Covariance(fqps, fs, nil)
	v := stat.Variance(fqps, nil)
	alpha := c / v
	alphas = make([]float64, len(preds))
	for i := range alphas {
		alphas[i] = alpha
	}
	return alphas
}
