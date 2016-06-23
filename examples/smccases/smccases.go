package smccases

import (
	"encoding/json"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/btracey/stackmc"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/stat"
	"github.com/gonum/stat/distuv"
	"github.com/gonum/stat/samplemv"
)

var RescaledBraninHoo = func(x []float64) float64 {
	if len(x) != 2 {
		panic("branin only coded for length 2")
	}
	x1 := x[0]
	x2 := x[1]
	pi := math.Pi
	a := 1.0
	b := 5.1 / (4 * pi * pi)
	c := 5.0 / pi
	r := 6.0
	s := 10.0
	t := 1 / (8.0 * pi)
	first := (x2 - b*x1*x1 + c*x1 - r)
	v := a*first*first + s*(1-t)*math.Cos(x1) + s

	if x[0] < -5 {
		v += math.Exp(5 - x[0])
	}
	if x[0] > 10 {
		v += math.Exp(x[0] - 10)
	}
	if x[1] < 0 {
		v += math.Exp(-x[1])
	}
	if x[1] > 15 {
		v += math.Exp(x[1] - 15)
	}
	return v
}

var quickDist1 = distuv.Normal{Mu: 1, Sigma: 3}
var quickDist2 = distuv.Normal{Mu: -1, Sigma: 5}
var quickDistW = 0.3

type QuickDistFunction struct{}

func (q QuickDistFunction) Func(x []float64) float64 {
	if len(x) != 1 {
		panic("bad size")
	}
	return x[0] * x[0]
}

func (q QuickDistFunction) Integrable(d stackmc.DistPredictor) bool {
	return false
}

func (q QuickDistFunction) ExpectedValue(d stackmc.DistPredictor) float64 {
	panic("not here")
	return math.NaN()
}

type QuickDistSampler struct{}

func (q QuickDistSampler) Sample(x *mat64.Dense) {
	nSamples, dim := x.Dims()
	if dim != 1 {
		panic("bad dim")
	}
	prand := func() float64 {
		if rand.Float64() < quickDistW {
			return quickDist1.Rand()
		}
		return quickDist2.Rand()
	}
	for i := 0; i < nSamples; i++ {
		x.Set(i, 0, prand())
	}
}

func (q QuickDistSampler) Prob(x []float64) float64 {
	if len(x) != 1 {
		panic("wrong size")
	}
	return quickDistW*quickDist1.Prob(x[0]) + (1-quickDistW)*quickDist2.Prob(x[0])
}

func (q QuickDistSampler) LogProb(x []float64) float64 {
	return math.Log(q.Prob(x))
}

type BraninHooTarget struct{}

func (b BraninHooTarget) LogProb(x []float64) float64 {
	temp := 50.0
	//temp := 100.0
	/*
		if x[0] < -5 || x[0] > 10 || x[1] < 0 || x[1] > 15 {
			// Out of bounds, impossible.
			return math.Inf(-1)
		}
	*/
	// x1 between -5 and 10, x2 between 0 and 15
	// Probability is proportional to e^(-H(sigma)/T)
	f := RescaledBraninHoo(x)
	return -f / temp
}

func GetBraninSampler() samplemv.MetropolisHastingser {
	sigma := mat64.NewSymDense(2, nil)
	sigma.SetSym(0, 0, 1)
	sigma.SetSym(1, 1, 1)

	prop, ok := samplemv.NewProposalNormal(sigma, nil)
	if !ok {
		panic("bad sigma")
	}
	initial := []float64{2.5, 7.5}
	mher := samplemv.MetropolisHastingser{
		Initial:  initial,
		Target:   BraninHooTarget{},
		Proposal: prop,
		Src:      nil,

		BurnIn: 10000,
		Rate:   1000,
	}
	return mher
}

type SqDistFrom struct {
	Center []float64
}

func (d SqDistFrom) Func(x []float64) float64 {
	return d.Dist(x)
}

func (d SqDistFrom) Integrable(stackmc.DistPredictor) bool {
	return false
}

func (d SqDistFrom) ExpectedValue(stackmc.DistPredictor) float64 {
	return math.NaN()
}

func (d SqDistFrom) Dist(x []float64) float64 {
	var dist float64
	for j, v := range x {
		d := v - d.Center[j]
		dist += d * d
	}
	return dist
}

var BraninCenter = []float64{math.Pi, 2.275}

// BraninEV returns the true ev.
// Function is expected distance from the middle well
func BraninEVDist() float64 {
	middle := BraninCenter

	// Generate a lot of samples
	nSamples := 100000
	batch := mat64.NewDense(nSamples, 2, nil)
	sampler := GetBraninSampler()
	sampler.Sample(batch)
	fmt.Println(batch.RawRowView(nSamples - 1))
	fmt.Println(BraninHooTarget{}.LogProb(batch.RawRowView(nSamples - 1)))

	// Find the expected distance
	var ev float64
	for i := 0; i < nSamples; i++ {
		x := batch.RawRowView(i)
		var dist float64
		for j, v := range x {
			d := v - middle[j]
			dist += d * d
		}
		//dist = math.Sqrt(dist)
		ev += dist
	}
	ev /= float64(nSamples)
	return ev
}

type RunData struct {
	CaseName         string
	MCName           string
	FitterNames      []string
	FitterPlotNames  []string
	StackMCNames     []string
	StackMCPlotNames []string
	TrueEV           float64
	Samples          []int
	EVMC             [][]float64
	EVFits           [][][]float64
	EVSmcs           [][][]float64
}

func SaveData(filename string, r RunData) error {
	if err := os.MkdirAll(filename, 0700); err != nil {
		return err
	}
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.MarshalIndent(r, "", "\t")
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	return err
}

func LoadData(filename string) (*RunData, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var r RunData
	decoder := json.NewDecoder(f)
	err = decoder.Decode(&r)
	return &r, nil
}

func MakePlots(savedir string, r *RunData) error {
	// Make plots. Make plots of the
	// 1) Expected value and error in the mean.
	// 2) Expected value and standard deviation.
	// 3) Expected Error and error in the mean
	// 4) Expected Error and standard deviation.

	truth := r.TrueEV
	evmc := r.EVMC

	// Data is first organized by sample, then by run. For the multiple ones,
	// each of these have a list of values. Reorganize to have the per-type
	// on the outside.
	nFits := len(r.EVFits[0][0])
	nSmc := len(r.EVSmcs[0][0])
	nSamples := len(evmc)
	nRuns := len(evmc[0])
	fits := make([][][]float64, nFits)
	for i := range fits {
		fits[i] = make([][]float64, nSamples)
		for j := range fits[i] {
			fits[i][j] = make([]float64, nRuns)
			for k := range fits[i][j] {
				fits[i][j][k] = r.EVFits[j][k][i]
			}
		}
	}
	smcs := make([][][]float64, nSmc)
	for i := range smcs {
		smcs[i] = make([][]float64, nSamples)
		for j := range smcs[i] {
			smcs[i][j] = make([]float64, nRuns)
			for k := range smcs[i][j] {
				smcs[i][j][k] = r.EVSmcs[j][k][i]
			}
		}
	}

	r.EVFits = fits
	r.EVSmcs = smcs

	plotfunc := func(x []float64) (m, l, h float64) {
		line := stat.Mean(x, nil) - truth
		std := stat.StdDev(x, nil)
		return line, std, std
	}
	xlab := "Number of Samples"
	if err := makeplot(savedir, "meanstd_"+r.CaseName, xlab, "Mean Error", true, false, plotfunc, r, false, false); err != nil {
		return err
	}

	plotfunc = func(x []float64) (m, l, h float64) {
		line := stat.Mean(x, nil) - truth
		std := stat.StdDev(x, nil)
		sterr := stat.StdErr(std, float64(len(x)))
		return line, sterr, sterr
	}
	if err := makeplot(savedir, "meaneim_"+r.CaseName, xlab, "Mean Error", true, false, plotfunc, r, false, false); err != nil {
		return err
	}

	/*
		plotfunc = func(x []float64) (m, l, h float64) {
			diff := make([]float64, len(x))
			for i := range x {
				diff[i] = math.Abs(x[i] - truth)
			}
			line := stat.Mean(diff, nil)
			std := stat.StdDev(diff, nil)
			return line, std, std
		}
		if err := makeplot(savedir, "sqerrstd_"+r.CaseName, xlab, "Mean Squared Error", true, true, plotfunc, r, true, false); err != nil {
			return err
		}
	*/

	plotfunc = func(x []float64) (m, l, h float64) {
		diff := make([]float64, len(x))
		for i := range x {
			diff[i] = (x[i] - truth) * (x[i] - truth)
		}
		line := stat.Mean(diff, nil)
		std := stat.StdDev(diff, nil)
		sterr := stat.StdErr(std, float64(len(x)))
		return line, sterr, sterr
	}
	if err := makeplot(savedir, "sqerreim_"+r.CaseName, xlab, "Mean Squared Error", true, true, plotfunc, r, true, false); err != nil {
		return err
	}

	return nil
}

func makeplot(dir, name, xlabel, ylabel string, logx, logy bool, plotfunc func([]float64) (m, l, h float64), r *RunData, left, top bool) error {
	p, err := plot.New()
	if err != nil {
		return err
	}

	datamc := getplotdata(plotfunc, logx, logy, r.Samples, r.EVMC)
	datafits := make([]YError, len(r.EVFits))
	for i := range datafits {
		datafits[i] = getplotdata(plotfunc, logx, logy, r.Samples, r.EVFits[i])
	}
	datasmcs := make([]YError, len(r.EVSmcs))
	fmt.Println("len ", len(r.EVSmcs))
	fmt.Println("len2", len(r.EVSmcs[0]))
	for i := range datasmcs {
		datasmcs[i] = getplotdata(plotfunc, logx, logy, r.Samples, r.EVSmcs[i])
	}
	{
		line, err := plotter.NewLine(datamc.XYs)
		if err != nil {
			return err
		}
		line.LineStyle.Color = mccolors[0]
		p.Add(line)
		p.Legend.Add(r.MCName, line)
		bars, err := plotter.NewYErrorBars(datamc)
		if err != nil {
			return err
		}
		bars.LineStyle.Color = mccolors[0]
		p.Add(bars)
	}

	for i := range datafits {
		line, err := plotter.NewLine(datafits[i].XYs)
		if err != nil {
			return err
		}
		line.LineStyle.Color = fitcolors[i]
		p.Add(line)
		p.Legend.Add(r.FitterPlotNames[i], line)
		bars, err := plotter.NewYErrorBars(datafits[i])
		if err != nil {
			return err
		}
		bars.LineStyle.Color = fitcolors[i]
		p.Add(bars)
	}

	for i := range datasmcs {
		line, err := plotter.NewLine(datasmcs[i].XYs)
		if err != nil {
			return err
		}
		line.LineStyle.Color = smccolors[i]
		p.Add(line)
		p.Legend.Add(r.StackMCPlotNames[i], line)
		bars, err := plotter.NewYErrorBars(datasmcs[i])
		if err != nil {
			return err
		}
		bars.LineStyle.Color = smccolors[i]
		p.Add(bars)
	}

	/*
		bars, err := plotter.NewYErrorBars(data)
		if err != nil {
			return err
		}
		bars.LineStyle.Color = mccolors[0]
		p.Add(line)
		p.Add(bars)
		p.Legend.Add(r.MCName)
	*/
	if logx {
		p.X.Scale = plot.LogScale{}
	}
	if logy {
		p.Y.Scale = plot.LogScale{}
	}
	p.X.Label.Text = xlabel
	p.Y.Label.Text = ylabel

	fullname := filepath.Join(dir, name)
	p.Legend.Left = left
	p.Legend.Top = top
	fmt.Println("presave")
	err = p.Save(4*vg.Inch, 3*vg.Inch, fullname+".pdf")
	if err != nil {
		return err
	}
	fmt.Println("postsave")
	return nil
}

func getplotdata(plotfunc func([]float64) (m, l, c float64), logx, logy bool, samples []int, data [][]float64) (y YError) {
	nSamples := len(samples)
	meanpts := make(plotter.XYs, nSamples)
	errpts := make(plotter.Errors, nSamples)
	for i := range meanpts {
		m, l, c := plotfunc(data[i])
		meanpts[i].X = float64(samples[i])
		meanpts[i].Y = m
		errpts[i].Low = l
		errpts[i].High = c
	}
	return YError{meanpts, plotter.YErrors(errpts)}
}

type YError struct {
	plotter.XYs
	plotter.YErrors
}

var mccolors = []color.Color{
	color.RGBA{15, 126, 18, 255},
}

var fitcolors = []color.Color{
	color.RGBA{252, 13, 27, 255},
}

func init() {
	//rand.New(0)
	rnd := rand.New(rand.NewSource(1))
	for i := 0; i < 100; i++ {
		red := uint8(rnd.Intn(255))
		green := uint8(rnd.Intn(255))
		blue := uint8(rnd.Intn(255))
		randcol := color.RGBA{red, green, blue, 255}
		smccolors = append(smccolors, randcol)
	}
}

// make a color list
var smccolors = []color.Color{
	color.RGBA{11, 36, 255, 255},
	color.RGBA{11, 26, 11, 255},
	color.RGBA{51, 246, 241, 255},
}

//func
