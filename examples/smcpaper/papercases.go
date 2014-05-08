package main

import (
	"encoding/json"
	"flag"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/btracey/stackmc"
	"github.com/btracey/stackmc/examples/smcpaper/helper"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

var gopath string

func init() {
	mat64.Register(goblas.Blas{})
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("need gopath")
	}
}

var defaultNumRuns = 2000

func main() {
	//defer profile.Start(profile.CPUProfile).Stop()
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) == 1 {
		log.Fatal("Must specify case name")
	}

	var casename string
	flag.StringVar(&casename, "case", "", "which case to run")
	var setNumDim int
	flag.IntVar(&setNumDim, "dim", -1, "how many dimensions in the problem")
	var setNumRuns int
	flag.IntVar(&setNumRuns, "runs", -1, "how many runs")
	flag.Parse()

	generator, sampSlice, nRuns, trueEv, nDim := GetRunDetails(casename, setNumDim)
	if nDim < 0 {
		log.Fatal("nDim not set")
	}
	if setNumRuns != -1 {
		nRuns = setNumRuns
	}

	log.Println("sample slice is ", sampSlice)
	log.Println("number of runs is ", nRuns)
	results, err := helper.MonteCarlo(generator, sampSlice, nRuns)
	if err != nil {
		log.Fatal(err)
	}
	eims := helper.ErrorInMean(results, trueEv)

	dimDir := "dim_" + strconv.Itoa(nDim)
	runDir := "runs_" + strconv.Itoa(nRuns)
	filePath := filepath.Join(gopath, "results", "stackmc", casename, dimDir, runDir)

	err = os.MkdirAll(filePath, 0700)
	if err != nil {
		log.Fatal(err)
	}
	jsonF, err := os.Create(filepath.Join(filePath, "eim_json.txt"))
	if err != nil {
		log.Fatal(err)
	}
	defer jsonF.Close()

	// Need to save the EIMS as a json too so can use later along with samle
	jsonStruct := struct {
		EIMS      []helper.SmcMse
		SampSlice []int
		NumDim    int
		NumRuns   int
	}{
		eims,
		sampSlice,
		nDim,
		nRuns,
	}

	b, err := json.MarshalIndent(jsonStruct, "", "\t")
	if err != nil {
		log.Fatal(err)
	}
	jsonF.Write(b)

	helper.PrintMses(eims, sampSlice)
	filename := filepath.Join(filePath, "eim.pdf")

	helper.PlotEIM(eims, sampSlice, filename)
}

func GetRunDetails(casename string, nDimA int) (generator helper.Generator, sampSlice []int, nRuns int, ev float64, realNumDim int) {

	switch casename {
	default:
		log.Fatal("Unknown casename")
	case "rosenunif":

		numSamp := 8
		realNumDim = nDimA
		if nDimA == -1 {
			realNumDim = 10
		}

		minSamp := 3.5 * float64(realNumDim)
		maxSamp := 20 * float64(realNumDim)

		sampSlice = helper.SampleRange(numSamp, minSamp, maxSamp)

		mins := make([]float64, realNumDim)
		maxs := make([]float64, realNumDim)
		for i := range maxs {
			mins[i] = -3
			maxs[i] = 3
		}
		dist := stackmc.NewUniform(mins, maxs)

		generator = &helper.StandardKFold{
			Dist:     dist,
			Function: helper.Rosen,
			FitterGenerators: []func() stackmc.Fitter{
				func() stackmc.Fitter {
					return &stackmc.Polynomial{
						Order: 3,
						Dist:  dist,
					}
				},
			},
			NumFolds: 10,
			NumDim:   realNumDim,
		}
		nRuns = defaultNumRuns
		ev = 1924.0 * float64(realNumDim-1)
	case "friedmanartificial":
		numSamp := 8
		if nDimA != -1 {
			log.Fatal("artificial has a fixed number of dimensions")
		}
		sampSlice = helper.SampleRange(numSamp, 35, 200)

	}
	return generator, sampSlice, nRuns, ev, realNumDim
}
