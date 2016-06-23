package stackmc

import (
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

// Do one per function sample? Average ghat - gi

type ConstantAlpha struct {
	Alpha float64
}

func (c ConstantAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = make([]float64, len(predictions[0]))
		for j := range alphas[i] {
			alphas[i][j] = c.Alpha
		}
	}
	return alphas
}

// SingleAlpha assumes there is a single fixed g. This is like the original StackMC.
type SingleAlpha struct{}

func (SingleAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {

	nFitters := len(predictions[0])
	var totalAlpha int
	for i := range folds {
		totalAlpha += len(folds[i].Assess)
	}
	assessData := mat64.NewDense(totalAlpha, nFitters+1, nil)
	var count int
	for i := range folds {
		for _, v := range folds[i].Assess {
			// First column is the actual function values
			assessData.Set(count, 0, f[v])
			// Rest of the columns are the fitter predictions.
			idx := uniqueMaps[i][v]
			for k := 0; k < nFitters; k++ {
				assessData.Set(count, k+1, predictions[i][k][idx])
			}
			count++
		}
	}

	alpha := controlVariateAlpha(assessData)

	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = make([]float64, len(alpha))
		copy(alphas[i], alpha)
	}
	//fmt.Println("normal alpha", alpha)
	return alphas
}

// SingleAlphaWithEVCorr uses the subtraction of the EV of that fold.
type SingleAlphaWithEVCorr struct{}

func (SingleAlphaWithEVCorr) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {

	nFitters := len(predictions[0])
	var totalAlpha int
	for i := range folds {
		totalAlpha += len(folds[i].Assess)
	}
	assessData := mat64.NewDense(totalAlpha, nFitters+1, nil)
	var count int
	for i := range folds {
		for _, v := range folds[i].Assess {
			// First column is the actual function values
			assessData.Set(count, 0, f[v])
			// Rest of the columns are the fitter predictions.
			idx := uniqueMaps[i][v]
			for k := 0; k < nFitters; k++ {
				assessData.Set(count, k+1, predictions[i][k][idx]-evs[k][i])
			}
			count++
		}
	}

	alpha := controlVariateAlpha(assessData)

	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = make([]float64, len(alpha))
		copy(alphas[i], alpha)
	}

	return alphas
}

// FoldIndAlpha computes a unique alpha for each fold, but doesn't take into
// account the covariances in the alphas.
type FoldIndAlpha struct{}

func (FoldIndAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	nFitters := len(predictions[0])
	alphas := make([][]float64, len(folds))
	for i, fold := range folds {
		nAssess := len(fold.Assess)
		assessData := mat64.NewDense(nAssess, nFitters+1, nil)
		for j, v := range fold.Assess {
			// First column is the actual function values
			assessData.Set(j, 0, f[v])
			// Rest of the columns are the fitter predictions.
			idx := uniqueMaps[i][v]
			for k := 0; k < nFitters; k++ {
				assessData.Set(j, k+1, predictions[i][k][idx])
			}
		}
		alphas[i] = controlVariateAlpha(assessData)
	}
	return alphas
}

func controlVariateAlpha(assessData *mat64.Dense) []float64 {
	r, c := assessData.Dims()
	_ = r
	nFitters := c - 1 // first column is real data
	covmat := stat.CovarianceMatrix(nil, assessData, nil)
	// The optimal alpha is A\b where b is variance of f with the fitter (row/column Zero)
	// and A are the covariances among the fitters (1:end, 1:end)
	covarWithF := mat64.NewVector(nFitters, nil)
	for i := 0; i < nFitters; i++ {
		covarWithF.SetVec(i, covmat.At(0, i+1))
	}
	fitterCovar := covmat.ViewSquare(1, nFitters).(*mat64.SymDense)
	//fitterCovar := (covmat.View(1, 1, len(fitters), len(fitters))).(*mat64.Dense)

	alpha := make([]float64, nFitters)

	alphaVec := mat64.NewVector(len(alpha), alpha)
	err := alphaVec.SolveVec(fitterCovar, covarWithF)
	if err != nil {
		// TODO(btracey): Handle error
		panic("fitting error")
	}
	return alpha
}

type ExpectedErrorAlphaHeldInFhat struct{}

func (ExpectedErrorAlphaHeldInFhat) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	num := make([]float64, len(folds))
	den := make([]float64, len(folds))
	for i, fold := range folds {
		var avgg, avgf float64
		for _, v := range fold.Assess {
			avgf += f[v]
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			avgg += pred
		}

		var fhat float64
		for _, v := range fold.Train {
			fhat += f[v]
		}
		fhat /= float64(len(fold.Train))

		avgf /= float64(len(fold.Assess))
		avgg /= float64(len(fold.Assess))
		gmis := avgg - evs[0][i]
		num[i] = gmis * (avgf - fhat)
		den[i] = gmis * gmis
	}
	alpha := stat.Mean(num, nil) / stat.Mean(den, nil)

	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

type ExpectedErrorAlpha struct{}

func (ExpectedErrorAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	var fhat float64
	var c int
	for _, fold := range folds {
		for _, idx := range fold.Train {
			fhat += f[idx]
			c++
		}
	}
	fhat /= float64(c)
	_ = fhat

	alpha := IterativeAlpha{}.estimateAlpha(fhat, f, predictions, uniqueMaps, folds, evs)

	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

type CheaterAlpha struct {
	TrueEV float64
}

func (c CheaterAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	alpha := IterativeAlpha{}.estimateAlpha(c.TrueEV, f, predictions, uniqueMaps, folds, evs)

	//	fmt.Println("cheater alpha", alpha)

	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

// IterativeAlpha computes
type IterativeAlpha struct {
	AllInd bool
}

func (it IterativeAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	//fhat := it.startingFhat(f, folds)
	//alpha := it.estimateAlpha(fhat, f, predictions, uniqueMaps, folds, evs)
	//fmt.Println("fhat, alpha", fhat, alpha)

	alpha := it.estimateAlphaFBias0(3, f, predictions, uniqueMaps, folds, evs)
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas

	/*

		nA := 501
		alphaStart := make([]float64, nA)
		alphaEnd := make([]float64, nA)

		floats.Span(alphaStart, -0.5, 1.5)
		for i, alpha := range alphaStart {
			fhat := it.estimateFhat(alpha, f, predictions, uniqueMaps, folds, evs)
			var alpha float64
			switch it.AlphaEst {
			default:
				panic("unknown alpha est")
			case "sampbiaszero":
				alpha = it.estimateAlphaFBias0(fhat, f, predictions, uniqueMaps, folds, evs)
				//fmt.Println("alpha = ", alpha)
			}
			alphaEnd[i] = alpha
			//alphaEnd[i] = it.estimateAlpha(fhat, f, predictions, uniqueMaps, folds, evs)
		}
		for i, v := range alphaEnd {
			alphaEnd[i] = math.Abs(v - alphaStart[i])
		}
		//fmt.Println(alphaEnd)
		idx := floats.MinIdx(alphaEnd)
		alpha := alphaStart[idx]

		//alphasmc := SingleAlpha{}.ComputeAlpha(f, predictions, folds, uniqueMaps, evs)

		//alpha = (alpha + alphasmc[0][0]) / 2

		//fmt.Println("iterative alpha", alpha)


		alphas := make([][]float64, len(folds))
		for i := range alphas {
			alphas[i] = []float64{alpha}
		}
		return alphas
	*/
}

func (it IterativeAlpha) startingFhat(f []float64, folds []Fold) float64 {
	var fhat float64
	var c int
	for _, fold := range folds {
		for _, idx := range fold.Train {
			fhat += f[idx]
			c++
		}
	}
	fhat /= float64(c)
	return fhat
}

func (it IterativeAlpha) estimateAlphaFBias0(fhat float64, f []float64, predictions [][][]float64, uniqueMaps []map[int]int, folds []Fold, evs [][]float64) float64 {
	if it.AllInd {
		numg := make([]float64, 0)
		numf := make([]float64, 0)
		den := make([]float64, 0)
		for i, fold := range folds {
			for _, v := range fold.Assess {
				idx := uniqueMaps[i][v]
				pred := predictions[i][0][idx]
				gdiff := pred - evs[0][i]
				numg = append(numg, gdiff)
				numf = append(numf, f[v])
				den = append(den, gdiff*gdiff)
			}
		}
		alpha := stat.Covariance(numg, numf, nil) / stat.Mean(den, nil)
		return alpha
	}
	numg := make([]float64, len(folds))
	numf := make([]float64, len(folds))
	dense := make([]float64, len(folds))
	dendiff := make([]float64, len(folds))
	for i, fold := range folds {
		var avgg, avgf float64
		for _, v := range fold.Assess {
			avgf += f[v]
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			avgg += pred
		}
		avgf /= float64(len(fold.Assess))
		avgg /= float64(len(fold.Assess))
		ghat := evs[0][i]
		gmis := avgg - ghat
		numg[i] = gmis
		numf[i] = avgf
		dense[i] = gmis * gmis
		dendiff[i] = gmis
	}
	alpha := float64(len(numg)-1) / float64(len(numg)) * stat.Covariance(numg, numf, nil) / stat.Mean(dense, nil)
	return alpha
}

func (it IterativeAlpha) estimateAlpha(fhat float64, f []float64, predictions [][][]float64, uniqueMaps []map[int]int, folds []Fold, evs [][]float64) float64 {
	num := make([]float64, len(folds))
	den := make([]float64, len(folds))
	for i, fold := range folds {
		var avgg, avgf float64
		for _, v := range fold.Assess {
			avgf += f[v]
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			avgg += pred
		}
		avgf /= float64(len(fold.Assess))
		avgg /= float64(len(fold.Assess))
		gmis := avgg - evs[0][i]
		num[i] = gmis * (avgf - fhat)
		den[i] = gmis * gmis
	}
	alpha := stat.Mean(num, nil) / stat.Mean(den, nil)
	return alpha
}

func (it IterativeAlpha) estimateFhat(alpha float64, f []float64, predictions [][][]float64, uniqueMaps []map[int]int, folds []Fold, evs [][]float64) float64 {
	nFolds := len(folds)
	foldEVs := make([]float64, nFolds)
	for i := 0; i < nFolds; i++ {
		var ev float64
		// alpha * ghat + \sum_i f_i - alpha g_i
		ev += alpha * evs[0][i]

		var avgfmg float64
		for _, idx := range folds[i].Update {
			truth := f[idx]
			predIdx := uniqueMaps[i][idx]
			pred := predictions[i][0][predIdx]
			avgfmg = truth - alpha*pred
		}
		avgfmg /= float64(len(folds[i].Update))
		foldEVs[i] = ev
	}
	return stat.Mean(foldEVs, nil)
}

type ExpectedErrorAlphaAllOne struct {
	FHat float64
}

func (e ExpectedErrorAlphaAllOne) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	var fhat float64
	var c int
	for _, fold := range folds {
		for _, idx := range fold.Train {
			fhat += f[idx]
			c++
		}
	}
	fhat /= float64(c)
	if e.FHat != 0 {
		fhat = e.FHat
	}

	num := make([]float64, len(folds))
	den := make([]float64, len(folds))
	for i, fold := range folds {
		var avgg, avgf float64
		for _, v := range fold.Assess {
			avgf += f[v]
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			avgg += pred
		}
		avgf /= float64(len(fold.Assess))
		avgg /= float64(len(fold.Assess))
		gmis := avgg - evs[0][i]
		num = append(num, gmis*(avgf-fhat))
		den = append(den, gmis*gmis)
	}
	alpha := stat.Mean(num, nil) / stat.Mean(den, nil)
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

// FullFoldAlpha computes alpha by cov(ghat - avg_g, avg_f) / var(ghat - avg_g)
type FullFoldAlpha struct {
	WithBias bool
}

func (full FullFoldAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	fterm := make([]float64, len(folds))
	gterm := make([]float64, len(folds)) // ghat - avg_g
	for i := range folds {

		var avgf float64
		var avgg float64
		for _, v := range folds[i].Assess {
			avgf += f[v]
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			avgg += pred
		}
		avgf /= float64(len(folds[i].Assess))
		fterm[i] = avgf

		avgg /= float64(len(folds[i].Assess))
		//avgg = evs[0][i] - avgg
		avgg -= evs[0][i]
		gterm[i] = avgg
	}
	cov := stat.Covariance(fterm, gterm, nil)
	variance := stat.Variance(gterm, nil)
	bias := stat.Mean(gterm, nil)
	var alpha float64
	if full.WithBias {
		//fmt.Println("variance, biassq, cov", variance, bias*bias, cov)
		alpha = cov / (variance + bias*bias)
	} else {
		alpha = cov / variance
	}

	//fmt.Println("alpha = ", alpha)
	//fmt.Println(alpha*alpha*variance - 2*alpha*cov + alpha*alpha*bias*bias)
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

// Like above, but use individual estimates
type FullFoldAlphaInd struct {
	WithBias bool
}

func (full FullFoldAlphaInd) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) [][]float64 {
	fterm := make([]float64, 0)
	gterm := make([]float64, 0) // ghat - avg_g
	for i := range folds {
		for _, v := range folds[i].Assess {
			fterm = append(fterm, f[v])
			idx := uniqueMaps[i][v]
			pred := predictions[i][0][idx]
			gterm = append(gterm, pred-evs[0][i])
		}
	}
	mean := stat.Mean(gterm, nil)
	variance := stat.Variance(gterm, nil)
	cov := stat.Covariance(fterm, gterm, nil)
	var alpha float64
	if full.WithBias {
		alpha = cov / (variance + mean*mean)
	} else {
		alpha = cov / variance
	}
	alphas := make([][]float64, len(folds))
	for i := range alphas {
		alphas[i] = []float64{alpha}
	}
	return alphas
}

/*
type RandGAlpha struct {
	Individual bool
}

func (r RandGAlpha) ComputeAlpha(f []float64, predictions [][][]float64, folds []Fold, uniqueMaps []map[int]int, evs [][]float64) []float64 {
	// Choose the alpha that minimizes the empiracle variance across the folds.
	nFitters := len(predictions[0])
	if nFitters != 1 {
		// TODO(btracey): Allow multiple fitters.
		panic("not coded for mulitple fitters")
	}
	nFolds := len(folds)

	var avgF, gMinusGHat []float64

	if r.Individual {
		// Construct the per-variable vectors of ghat, f and g
		avgF = make([]float64, 0)
		gMinusGHat = make([]float64, 0)
		for i := range folds {
			for _, v := range folds[i].Assess {
				avgF = append(avgF, f[v])
				idx := uniqueMaps[i][v]
				pred := predictions[i][0][idx]
				gMinusGHat = append(gMinusGHat, pred-evs[0][i])
			}
		}
	} else {
		// Construct the per-fold vectors of ghat, sum f and sum g.
		avgF = make([]float64, nFolds)
		gMinusGHat = make([]float64, nFolds)
		for i := range folds {
			var sf float64
			var sg float64
			for _, v := range folds[i].Assess {
				sf += f[v]
				idx := uniqueMaps[i][v]
				sg += predictions[i][0][idx]
			}
			n := float64(len(folds[i].Assess))
			avgF[i] = sf / n
			gMinusGHat[i] = sg/n - evs[0][i]
		}
	}
	alpha := stat.Covariance(gMinusGHat, avgF, nil) / stat.Variance(gMinusGHat, nil)
	return []float64{alpha}
}
*/
