package stackmc

import (
	"fmt"
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

type DistFitter interface {
	FitDist(x mat64.Matrix, f, logp []float64, inds []int) DistPredictor
}

type DistPredictor interface {
	// TODO(btracey): Should be log prob
	LogProb(x []float64) float64
	//Integrable(fun Function) bool
	//ExpectedValue(fun Function) float64
	Sample(x *mat64.Dense)
}

type DistFunction interface {
	Func([]float64) float64
	Integrable(d DistPredictor) bool
	ExpectedValue(d DistPredictor) float64
}

func FitDistEV(fit DistFitter, fun DistFunction, x mat64.Matrix, f, logp []float64, inds []int, evMult float64, evMin int) float64 {
	pred := fit.FitDist(x, f, logp, inds)
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
func EstimateDist(fun DistFunction, x mat64.Matrix, f, logp []float64, normalized bool, fitters DistFitter, folds []Fold) (ev float64) {

	nSamples, dim := x.Dims()
	if len(f) != nSamples {
		panic(errLen)
	}
	if len(logp) != nSamples {
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
	trueLogP := make([][]float64, nFolds)
	trueF := make([][]float64, nFolds)

	updatePredictions := make([][]float64, nFolds)
	row := make([]float64, dim)
	for i, fold := range folds {
		trueLogP[i] = make([]float64, len(fold.Assess))
		trueF[i] = make([]float64, len(fold.Assess))
		predictors[i] = fitters.FitDist(x, f, logp, fold.Train)
		alphaPredictions[i] = make([]float64, len(fold.Assess))
		for j, idx := range fold.Assess {
			mat64.Row(row, idx, x)
			alphaPredictions[i][j] = predictors[i].LogProb(row)
			trueLogP[i][j] = logp[idx]
			trueF[i][j] = f[idx]
		}
		updatePredictions[i] = make([]float64, len(fold.Update))
		for j, idx := range fold.Update {
			mat64.Row(row, idx, x)
			updatePredictions[i][j] = predictors[i].LogProb(row)
		}

		if !fun.Integrable(predictors[i]) {
			// Predict the outcome.
			newsamp := nSamples * 100
			if newsamp < 10000 {
				newsamp = 10000
			}
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

	// Compute the normalization constant from Monte-Carloing \int_x (q / p) p dx.
	// This works even if p and q are unnormalized distributions.
	var logZs []float64
	if normalized {
		logZs = make([]float64, nFolds)
		for i := range logZs {
			logZs[i] = 0
		}
	} else {
		logZs = estimateLogZ(trueLogP, alphaPredictions)
	}

	// Use this z to change the scale of the predictions
	for i := range alphaPredictions {
		z := logZs[i]
		for j := range alphaPredictions[i] {
			alphaPredictions[i][j] -= z
		}
	}
	for i := range updatePredictions {
		z := logZs[i]
		for j := range updatePredictions[i] {
			updatePredictions[i][j] -= z
		}
	}

	// Compute Alpha
	alphas := computeAlphaDist(trueLogP, trueF, alphaPredictions)
	fmt.Println("alphas = ", alphas)

	// Estimate the EV.
	var evEst float64
	for i, fold := range folds {
		var corrEV float64
		for j, idx := range fold.Update {
			// f * (p - alpha * q) / p  = f * (1 - alpha * q / p)
			v := (1 - alphas[i]*math.Exp(updatePredictions[i][j]-logp[idx]))
			//fmt.Println(v)
			//fmt.Println("uppred", updatePredictions[i][j], "lp", logp[idx])
			v2 := f[idx] * v
			corrEV += v2
			//corrEV += f[idx] * (p[idx] - alphas[i]*updatePredictions[i][j]) / p[idx] -- pre log version
		}
		corrEV /= float64(len(fold.Update))
		evEst += alphas[i]*evs[i] + corrEV
	}
	evEst /= float64(len(folds))
	return evEst
}

// TODO(btracey): make these interfaces like the normal one so can easily exchange
// types.
// TODO(btracey):

// estimateZ estimates Zp / Zq from the held out data (the correction factor to
// the q samples).
//   \int_p q/p p dx = zp / zq
// where notation is overloaded.
func estimateLogZ(logp, preds [][]float64) (logzs []float64) {
	var ps, qs []float64
	for i := range logp {
		for j := range logp[i] {
			ps = append(ps, logp[i][j])
			qs = append(qs, preds[i][j])
		}
	}
	divs := make([]float64, len(ps))
	for i := range divs {
		divs[i] = qs[i] - ps[i]
	}
	logZ := floats.LogSumExp(divs)
	logZ -= math.Log(float64(len(ps)))
	logzs = make([]float64, len(logp))
	for i := range logzs {
		logzs[i] = logZ
	}
	return logzs
}

func computeAlphaDist(logp, f, preds [][]float64) (alphas []float64) {

	var fqps, fs []float64
	for i := range preds {
		for j := range preds[i] {
			fs = append(fs, f[i][j])
			fqp := f[i][j] * math.Exp(preds[i][j]-logp[i][j])
			fqps = append(fqps, fqp)
			/*
				fs = append(fs, 1)
				fqps = append(fqps, math.Exp(preds[i][j]-logp[i][j]))
			*/
		}
	}

	c := stat.Covariance(fqps, fs, nil)
	v := stat.Variance(fqps, nil)
	fmt.Println(c, v)
	alpha := c / v
	if v == 0 {
		// All the sample locations are the same, so just use the Monte Carlo
		// estimate.
		alpha = 0
	}
	alphas = make([]float64, len(preds))
	for i := range alphas {
		alphas[i] = alpha
	}
	return alphas
}
