package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/distribution"
	"github.com/btracey/stackmc/examples/smccases"
	"github.com/btracey/stackmc/fit"
	"github.com/btracey/stackmc/fold"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize/functions"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distmv"
	"github.com/gonum/stat/distuv"
	"github.com/gonum/stat/samplemv"
)

type LogProber interface {
	LogProb(x []float64) float64
}

type ProbDister interface {
	stackmc.Distribution
}

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath not set")
	}
}

func main() {
	//rand.Seed(time.Now().UnixNano())
	//runtime.GOMAXPROCS(1)

	kind := os.Args[1]
	trueEV, runSettings, problemSettings, SMCSettings := getRunSettings(kind)

	if problemSettings.MCPlotName == "" {
		panic("need to set mc plot name")
	}
	if problemSettings.EvalType == FitFunc {
		if len(problemSettings.Fitters) != len(problemSettings.FitPlotName) {
			panic("fit plot names do not match")
		}
	} else {
		if len(problemSettings.FitPlotName) != len(problemSettings.FitDistribution) {
			panic("fit plot names do not match")
		}
	}
	for i := range problemSettings.FitPlotName {
		if problemSettings.FitPlotName[i] == "" {
			panic("fit name empty")
		}
	}

	for i := range SMCSettings {
		if SMCSettings[i].PlotName == "" {
			panic("smc name empty")
		}
	}

	sampleVec := makeSampleVec(runSettings)

	evmcs, evfits, evsmcs := SMCRuns(runSettings, problemSettings, SMCSettings)

	_, _, _ = evmcs, evfits, evsmcs

	// Save the data here.

	// Compute expected squared errors and jazz.
	avgMC, avgEimMC := computeAvgDiff(evmcs, trueEV)
	avgFit, avgEimFit := computeAvgDiffVec(evfits, trueEV)
	avgSmc, avgEimSmc := computeAvgDiffVec(evsmcs, trueEV)

	mcAvgBounds := make([][2]float64, len(avgMC))
	fitAvgBounds := make([][][2]float64, len(avgFit))
	smcAvgBounds := make([][][2]float64, len(avgSmc))
	for i := range mcAvgBounds {
		mcAvgBounds[i][0] = avgMC[i] - 2*avgEimMC[i]
		mcAvgBounds[i][1] = avgMC[i] + 2*avgEimMC[i]

		fitAvgBounds[i] = make([][2]float64, len(avgFit[i]))
		for j := range avgFit[i] {
			fitAvgBounds[i][j][0] = avgFit[i][j] - 2*avgEimFit[i][j]
			fitAvgBounds[i][j][1] = avgFit[i][j] + 2*avgEimFit[i][j]
		}

		smcAvgBounds[i] = make([][2]float64, len(avgSmc[i]))
		for j := range avgSmc[i] {
			smcAvgBounds[i][j][0] = avgSmc[i][j] - 2*avgEimSmc[i][j]
			smcAvgBounds[i][j][1] = avgSmc[i][j] + 2*avgEimSmc[i][j]
		}
	}

	eeMC, eimMC := computeError(evmcs, trueEV)
	eeFit, eimFit := computeErrorVec(evfits, trueEV)
	eesmcs, eimSmc := computeErrorVec(evsmcs, trueEV)
	_, _, _ = eimMC, eimFit, eimSmc

	fmt.Println("True value", trueEV)

	fmt.Println("Min Average values")
	for i := range sampleVec {
		fmt.Println(sampleVec[i], mcAvgBounds[i], fitAvgBounds[i], smcAvgBounds[i])
	}

	fmt.Println("Expected squared errors")
	for i := range sampleVec {
		fmt.Println(sampleVec[i], eeMC[i], eeFit[i], eesmcs[i])
	}

	// Plot the data
	r := &smccases.RunData{
		//CaseName:         "test",
		//MCName:           "MC",
		//FitterNames:      []string{"Poly"},
		//FitterPlotNames:  []string{"Polynomial"},
		//StackMCNames:     []string{"SMC"},
		//StackMCPlotNames: []string{"StackMC", "blah"},
		TrueEV:  trueEV,
		Samples: sampleVec,
		EVMC:    evmcs,
		EVFits:  evfits,
		EVSmcs:  evsmcs,
	}
	r.CaseName = kind
	r.MCName = problemSettings.MCPlotName
	r.FitterPlotNames = make([]string, len(problemSettings.FitPlotName))
	for i := range r.FitterPlotNames {
		r.FitterPlotNames[i] = problemSettings.FitPlotName[i]
	}
	r.StackMCPlotNames = make([]string, len(SMCSettings))
	for i := range r.StackMCPlotNames {
		r.StackMCPlotNames[i] = SMCSettings[i].PlotName
	}

	loc := filepath.Join(gopath, "results", "stackmc", "nips2016", kind, "plots")
	err := os.MkdirAll(loc, 0700)
	if err != nil {
		log.Fatal(err)
	}
	err = smccases.MakePlots(loc, r)
	if err != nil {
		log.Fatal(err)
	}
}

/*
func saveData(name string, trueEV float64, evmcs [][]float64, evfits, evsmcs [][][]float64) {
	location := filepath.Join(gopath, "results", "stackmc", "nips2016", "data", name)

}
*/

func computeAvgDiff(evmcs [][]float64, truth float64) (ee, eim []float64) {
	n := len(evmcs)
	ee = make([]float64, n)
	eim = make([]float64, n)
	for i := range evmcs {
		diff := make([]float64, len(evmcs[i]))
		for j := range evmcs[i] {
			diff[j] = evmcs[i][j] - truth
		}
		ee[i] = stat.Mean(diff, nil)
		eestd := stat.StdDev(diff, nil)
		eim[i] = stat.StdErr(eestd, float64(len(evmcs[i])))
	}
	return ee, eim
}

func computeError(evmcs [][]float64, truth float64) (ee, eim []float64) {
	n := len(evmcs)
	ee = make([]float64, n)
	eim = make([]float64, n)
	for i := range evmcs {
		diff := make([]float64, len(evmcs[i]))
		for j := range evmcs[i] {
			//diff[j] = math.Abs(evmcs[i][j] - truth)
			diff[j] = (evmcs[i][j] - truth) * (evmcs[i][j] - truth)
		}
		ee[i] = stat.Mean(diff, nil)
		eestd := stat.StdDev(diff, nil)
		eim[i] = stat.StdErr(eestd, float64(len(evmcs[i])))
	}
	return ee, eim
}

func computeAvgDiffVec(evs [][][]float64, truth float64) (avg, eim [][]float64) {
	n := len(evs)
	ee := make([][]float64, n)
	eim = make([][]float64, n)
	for i := range evs {
		nRuns := len(evs[i])
		diff := make([][]float64, nRuns)
		for j := range diff {
			diff[j] = make([]float64, len(evs[i][0]))
		}
		for j := 0; j < nRuns; j++ {
			for k := 0; k < len(evs[i][j]); k++ {
				diff[j][k] = evs[i][j][k] - truth
			}
		}

		ee[i] = make([]float64, len(evs[i][0]))
		eim[i] = make([]float64, len(evs[i][0]))
		d := make([]float64, nRuns)
		for k := 0; k < len(evs[i][0]); k++ {
			for j := 0; j < nRuns; j++ {
				d[j] = diff[j][k]
			}
			ee[i][k] = stat.Mean(d, nil)
			eestd := stat.StdDev(d, nil)
			eim[i][k] = stat.StdErr(eestd, float64(nRuns))
		}
	}
	return ee, eim
}

func computeErrorVec(evs [][][]float64, truth float64) (ee, eim [][]float64) {
	n := len(evs)
	ee = make([][]float64, n)
	eim = make([][]float64, n)
	for i := range evs {
		nRuns := len(evs[i])
		diff := make([][]float64, nRuns)
		for j := range diff {
			diff[j] = make([]float64, len(evs[i][0]))
		}
		for j := 0; j < nRuns; j++ {
			for k := 0; k < len(evs[i][j]); k++ {
				//diff[j][k] = math.Abs(evs[i][j][k] - truth)
				diff[j][k] = (evs[i][j][k] - truth) * (evs[i][j][k] - truth)
			}
		}

		ee[i] = make([]float64, len(evs[i][0]))
		eim[i] = make([]float64, len(evs[i][0]))
		d := make([]float64, nRuns)
		for k := 0; k < len(evs[i][0]); k++ {
			for j := 0; j < nRuns; j++ {
				d[j] = diff[j][k]
			}
			ee[i][k] = stat.Mean(d, nil)
			eestd := stat.StdDev(d, nil)
			eim[i][k] = stat.StdErr(eestd, float64(nRuns))
		}
	}
	return ee, eim
}

func getUniform(dim int, min, max float64) distribution.Uniform {
	dist := distribution.Uniform{}
	for i := 0; i < dim; i++ {
		dist.Unifs = append(dist.Unifs, distuv.Uniform{Min: min, Max: max})
	}
	return dist
}

func getIndGauss(dim int, mu float64, sigma float64) distribution.IndependentGaussian {
	dist := distribution.IndependentGaussian{}
	for i := 0; i < dim; i++ {
		dist.Norms = append(dist.Norms, distuv.Normal{Mu: mu, Sigma: sigma})
	}
	return dist
}

func getRunSettings(kind string) (float64, RunSettings, ProblemSettings, []SMCSettings) {
	switch kind {
	default:
		panic("unknown case")
	case "quadunifpaper":
		dim := 1
		ps := ProblemSettings{
			Name:       "QuadUnif",
			MCPlotName: "MC",
			Dim:        dim,
			EvalType:   FitFunc,
			Function: func(x []float64) float64 {
				if len(x) != 1 {
					panic("bad size")
				}
				return (x[0] - 0.2) * (x[0] - 0.2)
			},
			FitEVMult:         -1,
			GenDistribution:   getUniform(dim, 0, 1),
			InputDistribution: getUniform(dim, 0, 1),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{1}},
			FitPlotName:       []string{"Polynomial"},
		}
		trueEv := 13.0 / 75
		rs := RunSettings{
			MinSamples: 4,
			MaxSamples: 50,
			NumSamples: 14,
			Runs:       2000,
		}
		folds := 2
		_ = folds
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		smcset := []SMCSettings{
			{
				Name:        "Kfold",
				PlotName:    "StackMC Original",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
		}
		return trueEv, rs, ps, smcset
	case "quadunif":
		dim := 1
		ps := ProblemSettings{
			Name:       "QuadUnif",
			MCPlotName: "MC",
			Dim:        dim,
			EvalType:   FitFunc,
			Function: func(x []float64) float64 {
				if len(x) != 1 {
					panic("bad size")
				}
				return (x[0] - 0.2) * (x[0] - 0.2)
			},
			FitEVMult:         -1,
			GenDistribution:   getUniform(dim, 0, 1),
			InputDistribution: getUniform(dim, 0, 1),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{1}},
			FitPlotName:       []string{"Polynomial"},
		}
		trueEv := 13.0 / 75
		rs := RunSettings{
			MinSamples: 4,
			MaxSamples: 50,
			NumSamples: 14,
			Runs:       2000,
		}
		folds := 2
		_ = folds
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		smcset := []SMCSettings{
			{
				Name:        "Kfold",
				PlotName:    "StackMC Original",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.MultiKFold{folds, 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter10",
					Folder:   fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMC Updated",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMC Updated LOOCV",
					Folder:   fold.MultiKFold{K: 10000, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{true},
						EstimateFitEV: -1,
					},
				},
			*/

			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "Kfold",
					PlotName:    "StackMC10",
					Folder:      fold.MultiKFold{folds, 10},
					SMCSettings: smcSettings,
				},
				{
					Name:     "Kfold",
					PlotName: "EEA10",
					Folder:   fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlpha{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "EEAAllOne",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "EEAAllOne",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "EEAAllOneBoot10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldGhat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldGhatBias",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatNoBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{false},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldFoldBias",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldFoldGhat10Bias",
					Folder:   fold.MultiKFold{folds, 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold10",
					PlotName:    "KFold10",
					Folder:      fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhat",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatIndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhat",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhatIndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10IndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10IndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: smcSettings,
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "quaduniflatinpaper":
		dim := 1
		ps := ProblemSettings{
			Name:       "QuadUnifLatin",
			MCPlotName: "Latin Hypercube",
			Dim:        dim,
			EvalType:   FitFunc,
			Function: func(x []float64) float64 {
				if len(x) != 1 {
					panic("bad size")
				}
				return (x[0] - 0.2) * (x[0] - 0.2)
			},
			FitEVMult: -1,
			//GenDistribution: getUniform(dim, 0, 1),

			GenDistribution: samplemv.LatinHypercuber{
				Q: getUniform(dim, 0, 1),
			},

			InputDistribution: getUniform(dim, 0, 1),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{1}},
			FitPlotName:       []string{"Polynomial"},
		}
		trueEv := 13.0 / 75
		rs := RunSettings{
			MinSamples: 10,
			MaxSamples: 50,
			NumSamples: 6,
			Runs:       500,
		}
		folds := 5
		_ = folds
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Uncorr.",
				Folder:   fold.KFoldBoot{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Uncorr. LOOCV",
				Folder:   fold.KFoldBoot{K: rs.MaxSamples, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
		}
		return trueEv, rs, ps, smcset
	case "quaduniflatin":
		dim := 1
		ps := ProblemSettings{
			Name:       "QuadUnifLatin",
			MCPlotName: "Latin Hypercube",
			Dim:        dim,
			EvalType:   FitFunc,
			Function: func(x []float64) float64 {
				if len(x) != 1 {
					panic("bad size")
				}
				return (x[0] - 0.2) * (x[0] - 0.2)
			},
			FitEVMult: -1,
			//GenDistribution: getUniform(dim, 0, 1),

			GenDistribution: samplemv.LatinHypercuber{
				Q: getUniform(dim, 0, 1),
			},

			InputDistribution: getUniform(dim, 0, 1),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{1}},
			FitPlotName:       []string{"Polynomial"},
		}
		trueEv := 13.0 / 75
		rs := RunSettings{
			MinSamples: 10,
			MaxSamples: 50,
			NumSamples: 6,
			Runs:       500,
		}
		folds := 5
		_ = folds
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Uncorr.",
				Folder:   fold.KFoldBoot{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Uncorr. LOOCV",
				Folder:   fold.KFoldBoot{K: rs.MaxSamples, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},

			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.MultiKFold{folds, 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter",
					Folder:   fold.MultiKFold{K: folds, Multi: 50},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "quadunifcontrolfunc":
		dim := 1
		ps := ProblemSettings{
			Name:       "QuadUnif",
			MCPlotName: "MC",
			Dim:        dim,
			EvalType:   FitFunc,
			Function: func(x []float64) float64 {
				if len(x) != 1 {
					panic("bad size")
				}
				return x[0] * x[0]
			},
			FitEVMult:         -1,
			GenDistribution:   getIndGauss(dim, 0, 1),
			InputDistribution: getIndGauss(dim, 0, 1),
			//Fitters:           []stackmc.Fitter{&fit.Polynomial{1}},
			//FitPlotName:       []string{"Polynomial"},
			Fitters: []stackmc.Fitter{
				&fit.ControlFunc{
					Kernel: CFKernel{
						Alpha: [2]float64{0.1, 1},
					},
					Noise: 1e-10,
				},
			},
			FitPlotName: []string{"ControlFunctional"},
		}
		trueEv := 1.0
		rs := RunSettings{
			MinSamples: 20,
			MaxSamples: 1000,
			NumSamples: 15,
			Runs:       2000,
		}
		folds := 2
		_ = folds
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		smcset := []SMCSettings{
			{
				Name:        "Kfold",
				PlotName:    "StackMC Original",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			/*
				{
					Name:     "Kfold",
					PlotName: "Recursive",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.SingleAlpha{},
						EstimateFitEV: -1,
						Corrector:     stackmc.StackMCRecursive{},
					},
				},
			*/
			{
				Name:     "Kfold",
				PlotName: "FitInner",
				Folder:   fold.KFold{folds},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.SingleAlpha{},
					EstimateFitEV: -1,
					Corrector:     stackmc.FitInner{},
				},
			},
			/*
				{
					Name:     "Kfold",
					PlotName: "FitInnerEach",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.SingleAlpha{},
						EstimateFitEV: -1,
						Corrector:     stackmc.FitInnerEach{},
					},
				},
				{
					Name:     "Kfold",
					PlotName: "FitInnerEachAll",
					Folder:   fold.KFoldUpdateAll{folds, true, false},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.SingleAlpha{},
						EstimateFitEV: -1,
						Corrector:     stackmc.FitInnerEach{},
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "StackMC Original",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    true,
						Concurrent:    0,
						AlphaComputer: stackmc.SingleAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:         "Kfold",
					PlotName:     "AlphaCorrect",
					ExtraSampler: getIndGauss(dim, 0, 1),
					Folder:       fold.KFoldAlphaCorrect{folds, 100, false},
					SMCSettings:  smcSettings,
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMC Updated",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:         "KfoldEEA",
					PlotName:     "StackMC Updated",
					ExtraSampler: getIndGauss(dim, 0, 1),
					Folder:       fold.KFoldAlphaCorrect{folds, 100, false},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:         "KfoldEEA",
					PlotName:     "StackMC Updated",
					ExtraSampler: getIndGauss(dim, 0, 1),
					Folder:       fold.KFoldAlphaCorrect{folds, 100, true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:         "KfoldEEA",
					PlotName:     "StackMC Updated",
					ExtraSampler: getIndGauss(dim, 0, 1),
					Folder:       fold.KFoldAlphaCorrect{folds, 100, true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    true,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{true},
						EstimateFitEV: -1,
					},
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "rosengauss", "rosengaussfold20", "rosengausslb30":
		dim := 10
		ps := ProblemSettings{
			Name:              "RosenGauss",
			MCPlotName:        "MC",
			FitPlotName:       []string{"Polynomial"},
			Dim:               dim,
			EvalType:          FitFunc,
			Function:          functions.ExtendedRosenbrock{}.Func,
			GenDistribution:   getIndGauss(dim, 0, 2),
			InputDistribution: getIndGauss(dim, 0, 2),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{3}},
			FitEVMult:         -1,
		}

		rs := RunSettings{
			MinSamples: 40,
			MaxSamples: 80,
			NumSamples: 6,
			Runs:       5000,
		}
		if kind == "rosengausslb30" {
			rs.MinSamples = 30
		}
		/*
			rs := RunSettings{
				MinSamples: 30,
				MaxSamples: 50,
				NumSamples: 4,
				Runs:       200,
			}
		*/

		trueEv := 5205.0 * float64(dim-1)

		folds := 5
		if kind == "rosengaussfold20" {
			folds = 20
		}
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:        "Kfold",
				PlotName:    "StackMC",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
			{
				Name:     "KfoldEEA",
				PlotName: "StackMCIterInd",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},

			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaHeldInFhat{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter10",
					Folder:   fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter100",
					Folder:   fold.MultiKFold{K: folds, Multi: 100},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{0},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEACheat10",
					Folder:   fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KfoldEEAOtherCheat",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC10",
					Folder:      fold.MultiKFold{folds, 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "StackMC10Bias",
					Folder:   fold.MultiKFold{folds, 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "EEA",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlpha{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "EEAAllOne",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "EEA10",
					Folder:   fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
			*/

			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhatBias",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatNoBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{false},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KfoldIndAlpha",
					PlotName: "StackMC",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold10",
					PlotName:    "KFold10",
					Folder:      fold.MultiKFold{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhat",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatIndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhat",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhatIndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10IndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10IndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: smcSettings,
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "rosengausslatin":
		dim := 10
		ps := ProblemSettings{
			Name:       "RosenGauss",
			MCPlotName: "Latin Hypercube",
			Dim:        dim,
			EvalType:   FitFunc,
			Function:   functions.ExtendedRosenbrock{}.Func,
			GenDistribution: samplemv.LatinHypercuber{
				Q: getIndGauss(dim, 0, 2),
			},
			InputDistribution: getIndGauss(dim, 0, 2),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{3}},
			FitPlotName:       []string{"Polynomial"},
			FitEVMult:         -1,
		}

		// 50 800 6 2000
		rs := RunSettings{
			MinSamples: 50,
			MaxSamples: 800,
			NumSamples: 8,
			Runs:       2000,
		}
		trueEv := 5205.0 * float64(dim-1)
		/*
			rs := RunSettings{
				MinSamples: 200,
				MaxSamples: 400,
				NumSamples: 3,
				Runs:       1000,
			}
		*/

		folds := 5
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot",
				Folder:   fold.MultiBootstrap{K: folds, Times: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot 10",
				Folder:   fold.MultiBootstrap{K: folds, Times: 10},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},

				{
					Name:     "KfoldEEA",
					PlotName: "StackMCCheat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCIter",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{false},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCCheat",
					Folder:   fold.MultiKFold{K: folds, Multi: 100},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCCheat",
					Folder:   fold.KFoldBoot{K: folds, Multi: 100},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.CheaterAlpha{trueEv},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.IterativeAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KfoldEEA",
					PlotName: "StackMCEEA",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldBootEEA",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldBootEEA10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlphaAllOne{FHat: trueEv},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "Kfold",
					PlotName:    "KFoldBoot10",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
			*/

			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "EEA",
					Folder:   fold.MultiKFold{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.ExpectedErrorAlpha{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat",
					Folder:   fold.BootstrapInAllOut{50},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "MultiBoot",
					PlotName:    "MultiBoot10",
					Folder:      fold.MultiBootstrap{K: folds, Times: 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat10",
					Folder:   fold.MultiKFold{folds, 100},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat10",
					Folder:   fold.KFold{folds, 100},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldAll",
					Folder:   fold.KFoldUpdateAll{folds, true, true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlpha{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhat",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "Kfold",
					PlotName: "KFoldFullGhatBias",
					Folder:   fold.KFold{folds},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatNoBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{false},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "KFoldBootWithGhat",
					PlotName:    "KFoldBoot",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "Kfold",
					PlotName:    "StackMC",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
					{
						Name:        "Kfold10",
						PlotName:    "KFold10",
						Folder:      fold.MultiKFold{K: folds, Multi: 10},
						SMCSettings: smcSettings,
					},
				{
					Name:        "MultiBoot",
					PlotName:    "MultiBoot10",
					Folder:      fold.MultiBootstrap{K: folds, Times: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootWithGhat",
					PlotName:    "KFoldBoot10",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhat",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhat",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatIndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhat",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat",
					PlotName:    "KFoldBootNoGhatIndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 1, IndAlpha: true},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat10",
					PlotName: "KFoldBootWithGhat10IndAlpha",
					Folder:   fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{},
						EstimateFitEV: -1,
					},
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10},
					SMCSettings: smcSettings,
				},
				{
					Name:        "KFoldBootNoGhat10",
					PlotName:    "KFoldBootNoGhat10IndAlpha",
					Folder:      fold.KFoldBoot{K: folds, Multi: 10, IndAlpha: true},
					SMCSettings: smcSettings,
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "rosenunif":
		dim := 10
		ps := ProblemSettings{
			Name:              "RosenUnif",
			MCPlotName:        "MC",
			Dim:               dim,
			EvalType:          FitFunc,
			Function:          functions.ExtendedRosenbrock{}.Func,
			GenDistribution:   getUniform(dim, -3, 3),
			InputDistribution: getUniform(dim, -3, 3),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{3}},
			FitPlotName:       []string{"Polynomial"},
			FitEVMult:         -1,
		}

		rs := RunSettings{
			MinSamples: 50,
			MaxSamples: 800,
			NumSamples: 6,
			Runs:       2000,
		}

		folds := 5
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		smcset := []SMCSettings{
			{
				Name:        "Kfold",
				PlotName:    "StackMC",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			{
				Name:     "Kfold",
				PlotName: "StackMCGHat",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				//Folder:   fold.KFold{folds},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.FullFoldAlphaInd{},
					EstimateFitEV: -1,
				},
			},
		}
		trueEv := 1924.0 * float64(dim-1)
		return trueEv, rs, ps, smcset
	case "rosenuniflatin":
		dim := 10
		ps := ProblemSettings{
			Name:       "RosenUnif_Latin",
			MCPlotName: "Latin Hypercube",
			Dim:        dim,
			EvalType:   FitFunc,
			Function:   functions.ExtendedRosenbrock{}.Func,
			GenDistribution: samplemv.LatinHypercuber{
				Q: getUniform(dim, -3, 3),
			},
			InputDistribution: getUniform(dim, -3, 3),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{3}},
			FitPlotName:       []string{"Polynomial"},
			FitEVMult:         -1,
		}

		rs := RunSettings{
			MinSamples: 50,
			MaxSamples: 800,
			NumSamples: 8,
			Runs:       2000,
		}

		folds := 5
		foldsf := float64(folds)
		_ = foldsf
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot",
				Folder:   fold.MultiBootstrap{K: folds, Times: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot 10",
				Folder:   fold.MultiBootstrap{K: folds, Times: 10},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
		}
		trueEv := 1924.0 * float64(dim-1)
		return trueEv, rs, ps, smcset
	case "rosenunifhalton", "rosengausshalton":
		dim := 10
		ps := ProblemSettings{
			Name:              "RosenUnifHalton",
			MCPlotName:        "Halton Sequence",
			Dim:               dim,
			EvalType:          FitFunc,
			Function:          functions.ExtendedRosenbrock{}.Func,
			InputDistribution: getUniform(dim, -3, 3),
			Fitters:           []stackmc.Fitter{&fit.Polynomial{3}},
			FitPlotName:       []string{"Polynomial"},
			FitEVMult:         -1,
		}
		switch kind {
		default:
			panic("unknown")
		case "rosenunifhalton":
			ps.GenDistribution = &HaltonSampler{Quantiler: getUniform(dim, -3, 3)}
			ps.InputDistribution = getUniform(dim, -3, 3)
		case "rosengausshalton":
			ps.GenDistribution = &HaltonSampler{Quantiler: getIndGauss(dim, 0, 2)}
			ps.InputDistribution = getIndGauss(dim, 0, 2)
		}

		f, err := os.Open("../matlab/matlabjson.json")
		if err != nil {
			log.Fatal(err)
		}
		decoder := json.NewDecoder(f)
		haltonData := make([][][][]float64, 0)
		err = decoder.Decode(&haltonData)
		if err != nil {
			log.Fatal(err)
		}
		f.Close()
		fmt.Println("data read")

		rs := RunSettings{
			MinSamples: 50,
			MaxSamples: 800,
			NumSamples: 8,
			Runs:       2000,

			Data: haltonData,
		}

		folds := 5
		//foldsf := float64(folds)
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: -1,
		}
		_ = smcSettings
		smcset := []SMCSettings{
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Updated",
				Folder:   fold.MultiKFold{K: folds, Multi: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot",
				Folder:   fold.MultiBootstrap{K: folds, Times: 1},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},
			{
				Name:     "KfoldEEA",
				PlotName: "StackMC Boot 10",
				Folder:   fold.MultiBootstrap{K: folds, Times: 10},
				SMCSettings: &stackmc.Settings{
					UpdateFull:    false,
					Concurrent:    0,
					AlphaComputer: stackmc.IterativeAlpha{true},
					EstimateFitEV: -1,
				},
			},

			/*

				{
					Name:        "Kfold",
					PlotName:    "SMC K-Fold",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatNoBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{false},
						EstimateFitEV: -1,
					},
				},
				{
					Name:     "KFoldBootWithGhat",
					PlotName: "KFoldBootWithGhatBias",
					Folder:   fold.KFoldBoot{K: folds, Multi: 1},
					SMCSettings: &stackmc.Settings{
						UpdateFull:    false,
						Concurrent:    0,
						AlphaComputer: stackmc.FullFoldAlphaInd{true},
						EstimateFitEV: -1,
					},
				},
			*/
			/*
				{
					Name:        "MultiBoot",
					PlotName:    "SMC Bootstrap 10",
					Folder:      fold.MultiBootstrap{folds, 10, false},
					SMCSettings: smcSettings,
				},
			*/
		}
		var trueEv float64
		switch kind {
		default:
			panic("unknown kind")
		case "rosenunifhalton":
			trueEv = 1924.0 * float64(dim-1)
		case "rosengausshalton":
			trueEv = 5205.0 * float64(dim-1)
		}
		return trueEv, rs, ps, smcset
	case "quickdist":
		dim := 1
		trueEv := 21.233383235947354
		sampler := smccases.QuickDistSampler{}
		ps := ProblemSettings{
			Name:      "QuickDist",
			Dim:       dim,
			EvalType:  FitDist,
			FitEVMult: 100,
			FitEVMin:  10000,

			GenDistribution: sampler,

			FitDistribution:        []stackmc.DistFitter{fit.Gaussian{}},
			DistFunction:           smccases.QuickDistFunction{},
			LogProber:              sampler,
			NormalizedDistribution: false,
		}
		rs := RunSettings{
			MinSamples: 10,
			MaxSamples: 100,
			NumSamples: 4,
			Runs:       100,
		}
		folds := 5
		foldsf := float64(folds)
		_ = foldsf
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: 100,
		}
		smcset := []SMCSettings{
			{
				Name:        "KFold",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			{
				Name:        "UseAll",
				Folder:      fold.All{},
				SMCSettings: smcSettings,
			},
			{
				Name:        "MultiBoot",
				Folder:      fold.MultiBootstrap{folds, 1, false},
				SMCSettings: smcSettings,
			},
			/*
				{
					Name:        "MultiBoot",
					Folder:      fold.MultiBootstrap{folds, 10, false},
					SMCSettings: smcSettings,
				},
				{
					Name:        "MultiKFold",
					Folder:      fold.MultiKFold{folds, 10},
					SMCSettings: smcSettings,
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "rastriginmcmc2":
		target := smccases.MCMCTarget{
			Temp: 100,
			Func: smccases.Rastrigin{}.Func,
		}
		dim := 2
		fun := smccases.SqDistFrom{make([]float64, dim)}
		initial := make([]float64, dim)
		for i := range initial {
			initial[i] = 1
		}
		burnin := 1000
		rate := 100
		jumpdist := 1 / float64(dim)
		sampler := smccases.GetSampler(initial, target, jumpdist, burnin, rate)
		//trueEv := smccases.SamplerEV(sampler, fun.Func, dim, 100000)
		//fmt.Println("trueEV is", trueEv)
		trueEv := 99.8057135230803
		ps := ProblemSettings{
			Name:       "RatrigenDist",
			MCPlotName: "MCMC",
			Dim:        dim,
			EvalType:   FitDist,
			FitEVMult:  100,
			FitEVMin:   10000,

			GenDistribution: sampler,

			FitDistribution:        []stackmc.DistFitter{fit.Gaussian{}},
			FitPlotName:            []string{"Gaussian Fit"},
			DistFunction:           fun,
			LogProber:              sampler.Target,
			NormalizedDistribution: false,
		}
		rs := RunSettings{
			MinSamples: 25,
			MaxSamples: 1000,
			NumSamples: 8,
			Runs:       2000,
		}
		folds := 5
		foldsf := float64(folds)
		_ = foldsf
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: 100,
		}
		smcset := []SMCSettings{
			{
				Name:        "KFold",
				PlotName:    "StackMC",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
			/*
				{
					Name:        "UseAll",
					Folder:      fold.All{},
					SMCSettings: smcSettings,
				},
				{
					Name:        "MultiBoot",
					Folder:      fold.MultiBootstrap{folds, 1, false},
					SMCSettings: smcSettings,
				},
			*/
			/*
				{
					Name:        "Kfold",
					Folder:      fold.KFold{folds},
					SMCSettings: smcSettings,
				},
			*/
		}
		return trueEv, rs, ps, smcset
	case "rastriginmcmc30":
		target := smccases.MCMCTarget{
			Temp: 100,
			Func: smccases.Rastrigin{}.Func,
		}
		dim := 30
		fun := smccases.SqDistFrom{make([]float64, dim)}
		initial := make([]float64, dim)
		for i := range initial {
			initial[i] = 0
		}
		burnin := 30000
		rate := 3000
		jumpdist := 1 / float64(dim)
		sampler := smccases.GetSampler(initial, target, jumpdist, burnin, rate)
		//trueEv := smccases.SamplerEV(sampler, fun.Func, dim, 100000)
		//fmt.Println("trueEV is", trueEv)
		trueEv := 1500.7054523044937
		ps := ProblemSettings{
			Name:       "RatrigenDist",
			MCPlotName: "MCMC",
			Dim:        dim,
			EvalType:   FitDist,
			FitEVMult:  100,
			FitEVMin:   10000,

			GenDistribution: sampler,

			FitDistribution:        []stackmc.DistFitter{fit.Gaussian{}},
			FitPlotName:            []string{"Gaussian Fit"},
			DistFunction:           fun,
			LogProber:              sampler.Target,
			NormalizedDistribution: false,
		}
		rs := RunSettings{
			MinSamples: 100,
			MaxSamples: 1000,
			NumSamples: 8,
			Runs:       2000,
		}
		folds := 5
		foldsf := float64(folds)
		_ = foldsf
		smcSettings := &stackmc.Settings{
			UpdateFull:    false,
			Concurrent:    0,
			AlphaComputer: stackmc.SingleAlpha{},
			EstimateFitEV: 100,
		}
		smcset := []SMCSettings{
			{
				Name:        "KFold",
				PlotName:    "StackMC",
				Folder:      fold.KFold{folds},
				SMCSettings: smcSettings,
			},
		}
		return trueEv, rs, ps, smcset
	}
}

type EvalType int

const (
	FitFunc EvalType = iota + 1
	FitDist
)

type RunSettings struct {
	MinSamples int
	MaxSamples int
	NumSamples int
	Runs       int

	Data [][][][]float64 // for halton
}

type ProblemSettings struct {
	Name        string
	Dim         int
	EvalType    EvalType
	FitEVMult   float64
	FitEVMin    int
	MCPlotName  string
	FitPlotName []string

	//
	// Kind used to generate the samples and as input to StackMC. Frequently
	// the same, but different in latin hypercube (or whatever).
	// Probably can replace this and just be smarter.
	GenDistribution   stackmc.Distribution
	InputDistribution ProbDister
	Fitters           []stackmc.Fitter
	Function          func([]float64) float64

	// For fitting a probability distribution
	FitDistribution        []stackmc.DistFitter
	LogProber              LogProber // for getting the value of p
	NormalizedDistribution bool
	DistFunction           stackmc.DistFunction
}

type SMCSettings struct {
	Name         string
	PlotName     string
	Folder       fold.AdvFolder
	ExtraSampler stackmc.Distribution // if folder needs extra samples

	SMCSettings *stackmc.Settings
}

func SMCRuns(runset RunSettings, p ProblemSettings, smcSettings []SMCSettings) ([][]float64, [][][]float64, [][][]float64) {
	nRuns := runset.Runs
	sampleVec := makeSampleVec(runset)

	evmcs := make([][]float64, len(sampleVec))
	evfits := make([][][]float64, len(sampleVec))
	evsmcs := make([][][]float64, len(sampleVec))
	for s, samples := range sampleVec {
		fmt.Println(s+1, "samples = ", samples)
		evmcs[s] = make([]float64, nRuns)
		evfits[s] = make([][]float64, nRuns)
		evsmcs[s] = make([][]float64, nRuns)
		for i := 0; i < nRuns; i++ {
			hs, ok := p.GenDistribution.(*HaltonSampler)
			if ok {
				hs.Data = runset.Data[s][i]
			}

			evmc, evfit, evsmc := EvaluateSMC(samples, p, smcSettings)

			evmcs[s][i] = evmc
			evfits[s][i] = evfit
			evsmcs[s][i] = evsmc
			if (i+1)%25 == 0 {
				fmt.Println("done = ", i+1)
			}
		}
	}
	return evmcs, evfits, evsmcs
}

func EvaluateSMC(nSamples int, p ProblemSettings, smcSettings []SMCSettings) (evmc float64, evfit []float64, evsmc []float64) {

	// Generate the samples
	samples := mat64.NewDense(nSamples, p.Dim, nil)
	p.GenDistribution.Sample(samples)
	allInds := make([]int, nSamples)
	for i := range allInds {
		allInds[i] = i
	}

	fs := make([]float64, nSamples)
	for i := range fs {
		if p.EvalType == FitFunc {
			fs[i] = p.Function(mat64.Row(nil, i, samples))
		} else {
			fs[i] = p.DistFunction.Func(mat64.Row(nil, i, samples))
		}
	}

	var ps []float64
	if p.EvalType == FitDist {
		ps = make([]float64, nSamples)
		for i := range ps {
			ps[i] = math.Exp(p.LogProber.LogProb(mat64.Row(nil, i, samples)))
		}
	}

	evmc = stackmc.MCExpectedValue(fs, allInds)

	var evfitters []float64
	if p.EvalType == FitFunc {
		nFitter := len(p.Fitters)
		evfitters = make([]float64, nFitter)
		for i := range p.Fitters {
			evfitters[i] = stackmc.FitExpectedValue(p.Fitters[i], p.InputDistribution, samples, fs, allInds, p.FitEVMult)
		}
	} else if p.EvalType == FitDist {
		nFitter := len(p.FitDistribution)
		evfitters = make([]float64, nFitter)
		for i := range p.FitDistribution {
			evfitters[i] = stackmc.FitDistEV(p.FitDistribution[i], p.DistFunction, samples, fs, ps, allInds, p.FitEVMult, p.FitEVMin)
		}
	}

	evSmcs := make([]float64, len(smcSettings))
	for i, smcSet := range smcSettings {
		// Get the folds and extra samples if necessary.
		newSamples, folds := smcSet.Folder.AdvFolds(nSamples, p.Dim, smcSet.ExtraSampler)
		var nNewSamples int
		if newSamples != nil {
			nNewSamples, _ = newSamples.Dims()
		}

		// Append the new samples and compute the new fs (and ps).
		totalSamples := nSamples + nNewSamples
		smcSamples := mat64.NewDense(nSamples+nNewSamples, p.Dim, nil)
		smcSamples.Copy(samples)
		smcFs := make([]float64, totalSamples)
		copy(smcFs, fs)
		smcPs := make([]float64, totalSamples)
		copy(smcPs, ps)
		for i := 0; i < nNewSamples; i++ {
			samp := newSamples.RawRowView(i)
			smcSamples.SetRow(i+nSamples, samp)
			smcFs[i+nSamples] = p.Function(samp)
			if p.EvalType == FitDist {
				smcPs[i+nSamples] = math.Exp(p.LogProber.LogProb(samp))
			}
		}

		var evsmc float64
		switch p.EvalType {
		default:
			panic("unknown eval type")
		case FitFunc:
			// Run StackMC with these settings and new values.
			evsmc = stackmc.Estimate(p.InputDistribution, smcSamples, smcFs, p.Fitters, folds, smcSettings[i].SMCSettings)
		case FitDist:
			if len(p.FitDistribution) != 1 {
				panic("multiple distribution fits not coded")
			}
			evsmc = stackmc.EstimateDist(p.DistFunction, smcSamples, smcFs, smcPs, p.NormalizedDistribution, p.FitDistribution[0], folds)
		}
		evSmcs[i] = evsmc
	}
	return evmc, evfitters, evSmcs
}

func makeSampleVec(runset RunSettings) []int {
	sv := make([]float64, runset.NumSamples)
	floats.LogSpan(sv, float64(runset.MinSamples), float64(runset.MaxSamples))

	s := make([]int, runset.NumSamples)
	for i, v := range sv {
		s[i] = int(floats.Round(v, 0))
	}
	var sf []int
	for i, v := range s {
		if i != 0 && v == s[i-1] {
			continue
		}
		sf = append(sf, v)
	}
	return sf
}

type HaltonSampler struct {
	Data      [][]float64
	Quantiler distmv.Quantiler
}

func (h HaltonSampler) Sample(m *mat64.Dense) {
	r, c := m.Dims()
	if r != len(h.Data) {
		fmt.Println("r", r, len(h.Data))
		panic("bad size")
	}
	if c != len(h.Data[0]) {
		fmt.Println("c", c, len(h.Data[0]))
		panic("bad size")
	}

	pt := make([]float64, c)
	for i := range h.Data {
		h.Quantiler.Quantile(pt, h.Data[i])
		m.SetRow(i, pt)
	}
}

// Kernel used in Control Functionals for Monte Carlo integration.
type CFKernel struct {
	Alpha [2]float64
}

func (c CFKernel) Distance(x, y float64) float64 {
	//a1 := c.Alpha[0]
	a2 := c.Alpha[1]
	dist := x - y
	second := -dist * dist / (2 * a2 * a2)
	return math.Exp(second) * c.R(x) * c.R(y)
}

func (c CFKernel) R(x float64) float64 {
	return 1 / (1 + c.Alpha[0]*x*x)
}

func (c CFKernel) Deriv(x, y float64) float64 {
	a1 := c.Alpha[0]
	a2 := c.Alpha[1]
	ans1 := (-2*a1*c.R(x)*x - (1/a2)*(1/a2)*(x-y)) * c.Distance(x, y)
	/*
		h := 1e-6
		d1 := c.Distance(x+h, y)
		d2 := c.Distance(x-h, y)
		ans2 := (d1 - d2) / (2 * h)
		fmt.Println(ans1, ans2)
	*/
	return ans1
}

func (c CFKernel) Hessian(x, y float64) float64 {
	a1 := c.Alpha[0]
	a2 := c.Alpha[1]
	ans1 := (4*a1*a1*x*y*c.R(x)*c.R(y) + 2*a1*(1/a2)*(1/a2)*(x-y)*y*c.R(y) +
		(1/a2)*(1/a2) - 2*a1*(1/a2)*(1/a2)*x*(x-y)*c.R(x) -
		(1/a2)*(1/a2)*(1/a2)*(1/a2)*(x-y)*(x-y)) * c.Distance(x, y)
	return ans1
	/*
		// d/dx' dx k(x,x')
		h := 1e-3
		d := c.Distance(x, y)
		d1 := c.Distance(x+h, y+h)
		d2 := c.Distance(x+h, y)
		d3 := c.Distance(x, y+h)
		d4 := c.Distance(x-h, y)
		d5 := c.Distance(x, y-h)
		d6 := c.Distance(x-h, y-h)

		ans2 := (d1 - d2 - d3 + 2*d - d4 - d5 + d6) / (2 * h * h)
		fmt.Println(ans1, ans2)
		return ans1
	*/
}

/*
func (c CFKernel) R(x float64) float64{
	return 1 + c.Alpha[0] * 0
}
*/
