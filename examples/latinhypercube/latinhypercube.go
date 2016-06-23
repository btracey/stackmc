package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"

	"github.com/btracey/myplot"
	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/distribution"
	"github.com/btracey/stackmc/fit"
	"github.com/btracey/stackmc/fold"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize/functions"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distuv"
	"github.com/gonum/stat/samplemv"
)

type SampleType int

const (
	Independent SampleType = iota
	LatinHyper
	BigLatinHyper
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	blas64.Use(cgo.Implementation{})
	//lapack64.Use(lcgo.Implementation{})
}

func main() {
	// Rosen unif, but generated samples from Latin Hypercube.
	// Compare correlated and uncorrelated alpha/correction samples
	nDim := 10
	nRuns := 2000
	nSampleVec := 8
	sampleType := LatinHyper
	sampleVec := make([]float64, nSampleVec)
	floats.LogSpan(sampleVec, 40, 800)

	trueEv := 1924.0 * float64(nDim-1)

	mcEvs := make([][]float64, len(sampleVec))
	fitEvs := make([][]float64, len(sampleVec))
	smcEvs := make([][]float64, len(sampleVec))
	uncorrEvs := make([][]float64, len(sampleVec))
	for i, nSamples := range sampleVec {
		mcEvs[i] = make([]float64, nRuns)
		fitEvs[i] = make([]float64, nRuns)
		smcEvs[i] = make([]float64, nRuns)
		uncorrEvs[i] = make([]float64, nRuns)
		for j := 0; j < nRuns; j++ {
			fmt.Println(i, j)
			mcev, fitev, stackmcev, uncorrev := evs(int(nSamples), nDim, sampleType)
			mcEvs[i][j] = mcev
			fitEvs[i][j] = fitev
			smcEvs[i][j] = stackmcev
			uncorrEvs[i][j] = uncorrev
		}
	}
	mcEse, mcEim := eseEim(mcEvs, trueEv)
	fitEse, fitEim := eseEim(fitEvs, trueEv)
	stackMCEse, stackmcEim := eseEim(smcEvs, trueEv)
	uncorrEse, uncorrEim := eseEim(uncorrEvs, trueEv)

	_ = mcEim
	_ = fitEim
	_ = stackmcEim
	_ = uncorrEim
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	scatMc, err := plotter.NewLine(myplot.VecXY{X: sampleVec, Y: mcEse})
	if err != nil {
		log.Fatal(err)
	}
	//scatMc.LineStyle.Color = color.RGBA{G: 255}
	scatFit, err := plotter.NewLine(myplot.VecXY{X: sampleVec, Y: fitEse})
	if err != nil {
		log.Fatal(err)
	}
	//scatFit.LineStyle.Color = color.RGBA{R: 255}
	scatSmc, err := plotter.NewLine(myplot.VecXY{X: sampleVec, Y: stackMCEse})
	if err != nil {
		log.Fatal(err)
	}
	//scatMc.LineStyle.Color = color.RGBA{B: 255}
	scatUncorr, err := plotter.NewLine(myplot.VecXY{X: sampleVec, Y: uncorrEse})
	if err != nil {
		log.Fatal(err)
	}
	//scatUncorr.LineStyle.Color = color.RGBA{G: 255, B: 255}
	plotutil.AddLinePoints(p, "Polynomial", scatFit, "MC", scatMc, "SMC Default", scatSmc, "Uncorrelated", scatUncorr)
	//p.Add(scatMc, scatFit, scatSmc, scatUncorr)
	p.X.Scale = plot.LogScale{}
	p.Y.Scale = plot.LogScale{}
	p.Legend.Top = true
	p.Legend.Left = false
	err = p.Save(4*vg.Inch, 4*vg.Inch, "uncorrerr.pdf")
	if err != nil {
		log.Fatal(err)
	}

	resultFile := "result.dat"
	f, err := os.Create(resultFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	//f.WriteString("\n")
	f.WriteString(fmt.Sprintln(sampleVec))
	f.WriteString("mcEse = " + fmt.Sprintln(mcEse))
	f.WriteString("mcEim = " + fmt.Sprintln(mcEim))
	f.WriteString("fitEse = " + fmt.Sprintln(fitEse))
	f.WriteString("fitEim = " + fmt.Sprintln(fitEim))
	f.WriteString("stackMCEse = " + fmt.Sprintln(stackMCEse))
	f.WriteString("stackMCEim = " + fmt.Sprintln(stackmcEim))
	f.WriteString("uncorrEse = " + fmt.Sprintln(uncorrEse))
	f.WriteString("uncorrEim = " + fmt.Sprintln(uncorrEim))
}

func eseEim(data [][]float64, trueEv float64) ([]float64, []float64) {
	eses := make([]float64, len(data))
	eims := make([]float64, len(data))
	for i := range data {
		se := make([]float64, len(data[i]))
		for j := range se {
			err := data[i][j] - trueEv
			se[j] = err * err
		}
		eses[i] = stat.Mean(se, nil)
		v := stat.StdDev(se, nil)
		eims[i] = stat.StdErr(v, float64(len(se)))
	}
	return eses, eims
}

func evs(nSamples, nDim int, sampleType SampleType) (mcev, fitev, smcev, uncorrev float64) {
	nFolds := 5
	dist := distribution.Uniform{}
	for i := 0; i < nDim; i++ {
		dist.Unifs = append(dist.Unifs, distuv.Uniform{Min: -3, Max: 3})
	}
	function := functions.ExtendedRosenbrock{}.Func

	training, testing := fold.Partition(nSamples, nFolds)

	// Now, make completely uncorrelated testing indices
	totalSamples := nSamples
	uncorrTesting := make([][]int, nFolds)
	for i := range uncorrTesting {
		uncorrTesting[i] = make([]int, len(testing[i]))
		for j := range uncorrTesting[i] {
			uncorrTesting[i][j] = totalSamples
			totalSamples++
		}
	}
	// Generate the latin hypercube samples
	x := mat64.NewDense(totalSamples, nDim, nil)
	samplemv.LatinHypercube(x.View(0, 0, nSamples, nDim).(*mat64.Dense), dist, nil)

	// Generate the new samples
	switch sampleType {
	case Independent:
		for i := nSamples; i < totalSamples; i++ {
			dist.Rand(x.RawRowView(i))
		}
	case LatinHyper:
		thisRow := nSamples
		for i := range uncorrTesting {
			new := mat64.NewDense(len(testing[i]), nDim, nil)
			samplemv.LatinHypercube(new, dist, nil)
			for j := 0; j < len(testing[i]); j++ {
				for k := 0; k < nDim; k++ {
					v := new.At(j, k)
					x.Set(thisRow, k, v)
				}
				thisRow++
			}
		}
	}

	// Generate the uncorrelated samples
	//sample.LatinHypercube(x.View(nSamples, 0, totalSamples-nSamples, nDim).(*mat64.Dense), totalSamples-nSamples, nDim, distribution, nil)

	// Evaluate the function
	fs := make([]float64, totalSamples)
	for i := range fs {
		fs[i] = function(x.RawRowView(i))
	}

	fitter := &fit.Polynomial{
		Order: 3,
		//Distribution: dist,
	}

	/*
		samples := &stackmc.Samples{
			X: x,
			F: fs,
		}
	*/

	normalInds := make([]int, nSamples)
	for i := range normalInds {
		normalInds[i] = i
	}

	var wg sync.WaitGroup
	wg.Add(4)
	go func() {
		defer wg.Done()
		mcev = stackmc.MCExpectedValue(samples, normalInds)
	}()
	go func() {
		defer wg.Done()
		var err error
		fitev, err = stackmc.FitExpectedValue(fitter, samples, normalInds)
		if err != nil {
			panic(err)
		}
	}()
	go func() {
		defer wg.Done()
		settings := &stackmc.Settings{
			Training: training,
			Alpha:    testing,
			Correct:  testing,
		}
		smcev = stackmc.Estimate([]stackmc.Fitter{fitter}, samples, settings)
	}()
	go func() {
		defer wg.Done()
		settings := &stackmc.Settings{
			Training: training,
			Alpha:    uncorrTesting,
			Correct:  uncorrTesting,
		}
		uncorrev = stackmc.Estimate([]stackmc.Fitter{fitter}, samples, settings)
	}()
	wg.Wait()
	return
}
