package helper

import "math"

func Rosen(x []float64) float64 {
	if len(x) < 2 {
		panic("must have more than 1 dimension")
	}
	var sum float64
	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) + 100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	return sum
}
