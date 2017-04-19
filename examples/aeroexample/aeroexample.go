package aeroexample

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/samplemv"
)

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath must be set")
	}
}

var _ samplemv.Sampler = &OpenAeroSampler{}

type OpenAeroSampler struct {
	// Maybe add a case string eventually.
	Casename string

	// Maybe add with replacement
	x *mat64.Dense
	f []float64
}

// LoadAll must be called beforehand
func (o *OpenAeroSampler) LoadAll() error {
	avail := o.AvailableSamples()
	dim := o.SampleDim()
	o.x = mat64.NewDense(avail, dim, nil)
	o.f = make([]float64, avail)
	for i := 0; i < avail; i++ {
		f, err := o.loadSample(i, o.x.RawRowView(i))
		if err != nil {
			return err
		}
		o.f[i] = f
	}
	return nil
}

func (o *OpenAeroSampler) Func(x []float64) float64 {
	nSamples, dim := o.x.Dims()
	if len(x) != dim {
		panic("bad dim for function")
	}
	for i := 0; i < nSamples; i++ {
		x2 := o.x.RawRowView(i)
		if floats.Same(x, x2) {
			return o.f[i]
		}
	}
	panic("unknown x value")
}

func (o *OpenAeroSampler) Sample(batch *mat64.Dense) {
	nSamples, nDim := batch.Dims()
	if nSamples > o.AvailableSamples() {
		panic("more samples requested than available")
	}
	if nDim != o.SampleDim() {
		panic("wrong problem dimension")
	}
	avail := o.AvailableSamples()
	for i := 0; i < nSamples; i++ {
		idx := rand.Intn(avail)
		copy(batch.RawRowView(i), o.x.RawRowView(idx))
	}
}

// How many samples have been run
func (o *OpenAeroSampler) AvailableSamples() int {
	switch o.Casename {
	default:
		panic("unknown case")
	case "init_opt_uncertainty":
		return 9700
	}
}

// SampleDim is the dimension of the problem
func (o *OpenAeroSampler) SampleDim() int {
	switch o.Casename {
	default:
		panic("unknown case")
	case "init_opt_uncertainty":
		return 10
	}
}

// EmpiricalEV evaluates the empirical expected value from all the avaliable samples.
func (o *OpenAeroSampler) EmpiricalEV() float64 {
	var ev float64
	for _, v := range o.f {
		ev += v
	}
	return ev / float64(len(o.f))
}

func (o *OpenAeroSampler) basePath() string {
	return filepath.Join(gopath, "data", "sfi", "stackmc", "aeroopt", o.Casename)
}

func (o *OpenAeroSampler) loadSample(idx int, x []float64) (float64, error) {
	base := o.basePath()
	filename := filepath.Join(base, "result_"+strconv.Itoa(idx))
	f, err := os.Open(filename)
	defer f.Close()
	if err != nil {
		return math.NaN(), err
	}

	var result Result
	decoder := json.NewDecoder(f)
	err = decoder.Decode(&result)
	if err != nil {
		return math.NaN(), err
	}
	copy(x, result.RVs)
	return result.Result, nil
}

type Result struct {
	RVs    []float64
	Result float64
}
