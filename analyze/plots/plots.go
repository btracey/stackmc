package main

import (
	"log"
	"path/filepath"

	"github.com/btracey/stackmc/analyze"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
)

/*
var results = `
[40 50 63 80 100]
E[SMC], E[MC], E[Fit], ESE[SMC], ESE[MC], ESE[Fit]
452.66744886536014 479.3298631488143 483.98349709748896 4.582105352748443 6.356326033322592 4.451636907674727
450.647898838842 472.3772132845666 472.721759542206 3.776536810321097 5.9331781163258555 3.772139710424339
446.3135773228477 457.8000027195429 466.6817075673451 3.213728051434626 5.501244796607891 3.2105798652459923
437.8021326698828 442.9606114426409 451.97835504664295 2.7852612426664973 4.961821608997532 2.8655569825092555
438.015622393875 447.62269093895225 449.8264312183312 2.469450308806222 4.7220189512162785 2.536774180672959`
*/

var results = `
457.3740998129405 487.1452029774794 484.5657907244021 2.2530946951614768 3.1382770571455296 2.1896888920936775
450.43690211661004 471.2311815487906 472.21939738260704 1.9177793417131912 2.955860549471133 1.9077512152417744
445.94343188598094 461.87787058367536 460.9260359887097 1.6387281185645666 2.742704944125108 1.6432136627504343
440.05592810170106 448.3095370336756 451.4675856223241 1.3937035144242254 2.512746392153784 1.4102426660054492
441.4509616768799 443.8651232638767 447.96778572707643 1.2015908527741122 2.336110226894804 1.2341706264096997`

func main() {
	results := make([]analyze.AverageResult, 5)
	results[0].Samples = 40
	results[1].Samples = 50
	results[2].Samples = 63
	results[3].Samples = 80
	results[4].Samples = 100

	results[0].SmcEim = 457.3740998129405
	results[1].SmcEim = 450.43690211661004
	results[2].SmcEim = 445.94343188598094
	results[3].SmcEim = 440.05592810170106
	results[4].SmcEim = 441.4509616768799
	results[0].McEim = 487.1452029774794
	results[1].McEim = 471.2311815487906
	results[2].McEim = 461.87787058367536
	results[3].McEim = 448.3095370336756
	results[4].McEim = 443.8651232638767
	results[0].FitEim = []float64{484.5657907244021}
	results[1].FitEim = []float64{472.21939738260704}
	results[2].FitEim = []float64{460.9260359887097}
	results[3].FitEim = []float64{451.4675856223241}
	results[4].FitEim = []float64{447.96778572707643}

	results[0].SmcExpErr = 2.2530946951614768
	results[1].SmcExpErr = 1.9177793417131912
	results[2].SmcExpErr = 1.6387281185645666
	results[3].SmcExpErr = 1.3937035144242254
	results[4].SmcExpErr = 1.2015908527741122
	results[0].McExpErr = 3.1382770571455296
	results[1].McExpErr = 2.955860549471133
	results[2].McExpErr = 2.742704944125108
	results[3].McExpErr = 2.512746392153784
	results[4].McExpErr = 2.336110226894804
	results[0].FitExpErr = []float64{2.1896888920936775}
	results[1].FitExpErr = []float64{1.9077512152417744}
	results[2].FitExpErr = []float64{1.6432136627504343}
	results[3].FitExpErr = []float64{1.4102426660054492}
	results[4].FitExpErr = []float64{1.2341706264096997}

	/*
		results[0].SmcEim = 452.66744886536014
		results[1].SmcEim = 450.647898838842
		results[2].SmcEim = 446.3135773228477
		results[3].SmcEim = 437.8021326698828
		results[4].SmcEim = 438.015622393875
		results[0].McEim = 479.3298631488143
		results[1].McEim = 472.3772132845666
		results[2].McEim = 457.8000027195429
		results[3].McEim = 442.9606114426409
		results[4].McEim = 447.62269093895225
		results[0].FitEim = []float64{483.98349709748896}
		results[1].FitEim = []float64{472.721759542206}
		results[2].FitEim = []float64{466.6817075673451}
		results[3].FitEim = []float64{451.97835504664295}
		results[4].FitEim = []float64{449.8264312183312}

		results[0].SmcExpErr = 4.582105352748443
		results[1].SmcExpErr = 3.776536810321097
		results[2].SmcExpErr = 3.213728051434626
		results[3].SmcExpErr = 2.7852612426664973
		results[4].SmcExpErr = 2.469450308806222
		results[0].McExpErr = 6.356326033322592
		results[1].McExpErr = 5.9331781163258555
		results[2].McExpErr = 5.501244796607891
		results[3].McExpErr = 4.961821608997532
		results[4].McExpErr = 4.7220189512162785
		results[0].FitExpErr = []float64{4.451636907674727}
		results[1].FitExpErr = []float64{3.772139710424339}
		results[2].FitExpErr = []float64{3.2105798652459923}
		results[3].FitExpErr = []float64{2.8655569825092555}
		results[4].FitExpErr = []float64{2.536774180672959}
	*/

	settings := Settings{
		Title:    "",
		MCName:   "MCMC",
		FitNames: []string{"Poly"},
	}
	esePlot("", "mcmcerr", results, settings)
}

type Settings struct {
	Title    string
	MCName   string
	FitNames []string
}

type yerrs struct {
	plotter.XYs
	plotter.YErrors
}

func esePlot(path, name string, results []analyze.AverageResult, settings Settings) {
	pointVec := make([]float64, len(results))
	for i, result := range results {
		pointVec[i] = float64(result.Samples)
	}

	smcMeans := make([]float64, len(results))
	smcEIMs := make([]float64, len(results))
	for i, result := range results {
		smcMeans[i] = result.SmcEim
		smcEIMs[i] = result.SmcExpErr
	}
	smcErrorBars, err := makeErrorBars(pointVec, smcMeans, smcEIMs)
	if err != nil {
		log.Fatal(err)
	}
	smcErrorBars.Color = plotutil.SoftColors[0]

	mcMeans := make([]float64, len(results))
	mcEIMs := make([]float64, len(results))
	for i, result := range results {
		mcMeans[i] = result.McEim
		mcEIMs[i] = result.McExpErr
	}
	mcErrorBars, err := makeErrorBars(pointVec, mcMeans, mcEIMs)
	if err != nil {
		log.Fatal(err)
	}
	mcErrorBars.Color = plotutil.SoftColors[1]

	if len(results[0].FitExpErr) != 1 {
		panic("only coded for one fitter")
	}
	fitMeans := make([]float64, len(results))
	fitEIMs := make([]float64, len(results))
	for i, result := range results {
		fitMeans[i] = result.FitEim[0]
		fitEIMs[i] = result.FitExpErr[0]
	}
	fitErrorBars, err := makeErrorBars(pointVec, fitMeans, fitEIMs)
	if err != nil {
		log.Fatal(err)
	}
	fitErrorBars.LineStyle.Color = plotutil.SoftColors[2]

	plt, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	plotutil.AddLines(plt, "SMC", smcErrorBars.XYs, settings.MCName, mcErrorBars.XYs, settings.FitNames[0], fitErrorBars.XYs)
	plotutil.AddErrorBars(plt, smcErrorBars, mcErrorBars, fitErrorBars)
	loc := filepath.Join(path, name+".pdf")
	plt.Legend.Top = true
	plt.Legend.Left = false

	plt.Title.Text = settings.Title
	plt.X.Label.Text = "Number of Samples"
	plt.Y.Label.Text = "Expected Error with Error in the Mean"
	err = plt.Save(4.48*vg.Inch, 3.37*vg.Inch, loc)
	if err != nil {
		log.Fatal(err)
	}
}

func makeErrorBars(pointVec, means, eims []float64) (*plotter.YErrorBars, error) {
	if len(pointVec) != len(means) {
		panic("slice length mismatch")
	}
	if len(means) != len(eims) {
		panic("slice length mismatch")
	}
	n := len(pointVec)
	// Make the SMC Bar plot
	xys := make(plotter.XYs, n)
	for i, v := range means {
		xys[i].X = pointVec[i]
		xys[i].Y = v
	}
	yErrors := make(plotter.YErrors, n)
	for i, v := range eims {
		yErrors[i].Low = v
		yErrors[i].High = v
	}
	yErrorBars, err := plotter.NewYErrorBars(yerrs{xys, yErrors})
	return yErrorBars, err
}
