package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"strconv"
	"time"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/analyze"
	"github.com/btracey/stackmc/distribution"
	"github.com/btracey/stackmc/fit"
	"github.com/btracey/stackmc/fold"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize/functions"
	"github.com/gonum/stat/distuv"
	"github.com/gonum/stat/samplemv"
)

type Example struct {
	Name          string
	Lb, Ub        int
	SampleVecSize int
	NRuns         int

	TrueEV float64

	Settings analyze.Settings
}

func mcmcRosen(dim int, updateFull bool, ac stackmc.AlphaComputer, folder fold.Folder) Example {
	e := Example{
		Name:          "Rosenbrock_mcmc_" + strconv.Itoa(dim),
		Lb:            int(20 * float64(dim)),
		Ub:            50 * dim,
		SampleVecSize: 5,
		NRuns:         16000,
	}
	width := 0.5
	sigma := mat64.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		sigma.SetSym(i, i, width)
	}
	proposal, ok := samplemv.NewProposalNormal(sigma, nil)
	if !ok {
		panic("bad proposal")
	}
	initial := make([]float64, dim)
	target := distribution.Uniform{}
	for i := 0; i < dim; i++ {
		target.Unifs = append(target.Unifs, distuv.Uniform{Min: -3, Max: 3})
	}
	distribution := samplemv.MetropolisHastingser{
		Initial:  initial,
		Target:   target,
		Proposal: proposal,
		Src:      nil,
		BurnIn:   50,
		Rate:     100,
	}

	fitter := &fit.Polynomial{
		Order: 3,
	}

	e.Settings = analyze.Settings{
		SettingsSMC: &stackmc.Settings{
			UpdateFull:    updateFull,
			AlphaComputer: ac,
			EstimateFitEV: 30,
		},
		Distribution: distribution,
		Fitters:      []stackmc.Fitter{fitter},

		Folder: folder,
		Func:   functions.ExtendedRosenbrock{}.Func,
		Dim:    dim,
	}
	e.TrueEV = 1924.0 * float64(dim-1)
	return e
}

func constructRosen(dim int, updateFull bool, ac stackmc.AlphaComputer, folder fold.Folder) Example {
	e := Example{
		Name:          "Rosenbrock_" + strconv.Itoa(dim),
		Lb:            int(3.2 * float64(dim)),
		Ub:            20 * dim,
		SampleVecSize: 8,
		NRuns:         2000,
	}
	distribution := distribution.Uniform{}
	for i := 0; i < dim; i++ {
		distribution.Unifs = append(distribution.Unifs, distuv.Uniform{Min: -3, Max: 3})
	}
	fitter := &fit.Polynomial{
		Order: 3,
	}
	e.Settings = analyze.Settings{
		SettingsSMC: &stackmc.Settings{
			UpdateFull:    updateFull,
			AlphaComputer: ac,
		},
		Distribution: distribution,
		Fitters:      []stackmc.Fitter{fitter},

		//Sampler: sample.IID{distribution},
		Folder: folder,
		Func:   functions.ExtendedRosenbrock{}.Func,
		Dim:    dim,
	}
	e.TrueEV = 1924.0 * float64(dim-1)
	return e
}

var (
	NDRosen Example
)

func main() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU() - 2)
	// e := constructRosen(10, false, stackmc.FixedGAlpha{}, fold.KFoldBootstrap{5, 1})
	//e := constructRosen(10, false, stackmc.RandGAlpha{true}, fold.KFoldBootstrap{5, 1})
	// e := constructRosen(10, false, stackmc.RandGAlpha{}, fold.KFoldBootstrap{10, 10})
	e := mcmcRosen(2, false, stackmc.FixedGAlpha{}, fold.KFoldBootstrap{5, 1})
	fmt.Println("truth = ", e.TrueEV)
	sampleVec := make([]int, e.SampleVecSize)
	analyze.RoundedLogSpan(sampleVec, e.Lb, e.Ub)
	results := analyze.SweepAverage(e.TrueEV, e.NRuns, sampleVec, e.Settings)
	fmt.Println(e.Name)
	fmt.Println("E[SMC], E[MC], E[Fit], ESE[SMC], ESE[MC], ESE[Fit]")
	for i := range results {
		fmt.Println(results[i].SmcExpErr, results[i].McExpErr, results[i].FitExpErr[0], results[i].SmcEim, results[i].McEim, results[i].FitEim[0])
	}

	// TODO(btracey): Create marshaling into a JSON struct
}
