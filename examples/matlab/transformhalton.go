package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

var samps = []int{50, 74, 110, 164, 244, 362, 538, 800}
var runs = 2000
var dim = 10

func main() {
	f, err := os.Open("halton10_matlaboutput.txt")
	if err != nil {
		log.Fatal(err)
	}

	// Read up until the first "ans"
	data := make([][][][]float64, len(samps))
	for i := range data {
		data[i] = make([][][]float64, runs)
		for j := range data[i] {
			data[i][j] = make([][]float64, samps[i])
		}
	}

	// Read until the first "ans"
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		str := scanner.Text()
		str = strings.TrimSpace(str)
		if len(str) < 3 {
			continue
		}
		if str[0:3] == "ans" {
			scanner.Scan()
			break
		}
	}

	// Read in the data.
	var str string
	// Read in the data samples
	for sidx, samp := range samps {
		fmt.Println("sidx")
		// Read in all of the runs
		for run := 0; run < runs; run++ {
			fmt.Println("run = ", run)
			for j := 0; j < samp; j++ {
				scanner.Scan()
				str = scanner.Text()
				strsfull := strings.Split(str, " ")
				var strs []string
				for _, s := range strsfull {
					if s != "" {
						strs = append(strs, s)
					}
				}

				for i := range strs {
					strs[i] = strings.TrimSpace(strs[i])
				}
				if len(strs) != dim {
					fmt.Println("len strs", len(strs))
					fmt.Println(str)
					log.Fatal("length of sample not dim")
				}
				pt := make([]float64, dim)
				for i := range pt {
					f, err := strconv.ParseFloat(strs[i], 64)
					if err != nil {
						log.Fatal(err)
					}
					pt[i] = f
				}
				data[sidx][run][j] = pt
			}
			// Read to the next sample
			scanner.Scan()
			scanner.Scan()
			scanner.Scan()
			scanner.Scan()
		}
	}
	f.Close()

	f, err = os.Create("matlabjson.json")
	if err != nil {
		log.Fatal(err)
	}
	b, err := json.MarshalIndent(data, "", "\t")
	_, err = f.Write(b)
	if err != nil {
		log.Fatal(err)
	}
	f.Close()
}
