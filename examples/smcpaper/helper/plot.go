package helper

import (
	"image/color"
	"math"
	"strconv"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
)

type errorBarPlotter struct {
	Locs []float64
	Eims []Eim
}

func (e errorBarPlotter) Len() int {
	return len(e.Eims)
}

func (e errorBarPlotter) XY(i int) (x, y float64) {
	return math.Log10(e.Locs[i]), math.Log10(e.Eims[i].MeanSquaredError)
}

func (e errorBarPlotter) YError(i int) (float64, float64) {
	plus2std := math.Log10(e.Eims[i].MeanSquaredError + 2*e.Eims[i].ErrorInMean)
	minus2std := math.Log10(e.Eims[i].MeanSquaredError - 2*e.Eims[i].ErrorInMean)
	mean := math.Log10(e.Eims[i].MeanSquaredError)
	return mean - minus2std, plus2std - mean
}

func PlotEIM(eims []SmcMse, nSampSlice []int, plotLoc string) error {
	plt, err := plot.New()
	if err != nil {
		return nil
	}
	nSamp := len(nSampSlice)

	xLocs := make([]float64, nSamp)
	for i := range nSampSlice {
		xLocs[i] = float64(nSampSlice[i])
	}

	mc := errorBarPlotter{}
	mc.Locs = xLocs
	mc.Eims = make([]Eim, nSamp)
	for i, eim := range eims {
		mc.Eims[i] = eim.Mc
	}
	mcErrBars, err := plotter.NewYErrorBars(mc)
	if err != nil {
		return err
	}

	mcLine, _, err := plotter.NewLinePoints(mc)
	if err != nil {
		return err
	}

	mcColor := color.RGBA{G: 255}
	mcLine.Color = mcColor
	mcErrBars.Color = mcColor

	smc := errorBarPlotter{}
	smc.Locs = xLocs
	smc.Eims = make([]Eim, nSamp)
	for i, eim := range eims {
		smc.Eims[i] = eim.StackMc
	}
	smcErrBars, err := plotter.NewYErrorBars(smc)
	if err != nil {
		return err
	}

	smcLine, _, err := plotter.NewLinePoints(smc)
	if err != nil {
		return err
	}
	smcColor := color.RGBA{B: 255}
	smcLine.Color = smcColor
	smcErrBars.Color = smcColor

	fitmc := make([]*plotter.YErrorBars, len(eims[0].Fitters))
	fitLines := make([]*plotter.Line, len(eims[0].Fitters))
	for j := range fitmc {
		ebp := errorBarPlotter{}
		ebp.Locs = xLocs
		ebp.Eims = make([]Eim, nSamp)
		for i, eim := range eims {
			ebp.Eims[i] = eim.Fitters[j]
		}
		fitmc[j], err = plotter.NewYErrorBars(ebp)
		if err != nil {
			return err
		}

		fitLines[j], _, err = plotter.NewLinePoints(ebp)
		if err != nil {
			return err
		}

		fitColor := color.RGBA{R: 255}
		fitmc[j].Color = fitColor
		fitLines[j].Color = fitColor
	}

	plotters := []plot.Plotter{mcErrBars, smcErrBars, mcLine, smcLine}
	for i := range fitmc {
		plotters = append(plotters, fitmc[i], fitLines[i])
	}

	plt.Add(plotters...)

	plt.Legend.Add("StackMC", smcLine)
	plt.Legend.Add("MonteCarlo", mcLine)
	for i := range fitmc {
		plt.Legend.Add("Fit_"+strconv.Itoa(i), fitLines[i])
	}
	plt.Legend.Left = true
	plt.Legend.Top = false
	plt.Save(4, 4, plotLoc)
	return nil
}
