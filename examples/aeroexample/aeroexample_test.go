package aeroexample

import (
	"fmt"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestAeroExample(t *testing.T) {
	oas := &OpenAeroSampler{Casename: "init_opt_uncertainty"}
	oas.LoadAll()

	nSamples := 100
	batch := mat64.NewDense(nSamples, oas.SampleDim(), nil)
	oas.Sample(batch)

	f := make([]float64, nSamples)
	for i := range f {
		f[i] = oas.Func(batch.RawRowView(i))
	}
	fmt.Println(f)

	fmt.Println(oas.EmpiricalEV())
}
