// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/textus/mat64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// todo
// calculate page rank of mixer stochastic tree searches

const (
	// VectorSize is the size of a vector
	VectorSize = InputSize * 4
	// ItemSize is the size of a row
	ItemSize = VectorSize + 1
	// Samples is the number of samples
	Samples = 8 * 1024
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Scale is the scale of the model
	Scale = 128
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed books/*
var Data embed.FS

var (
	// FlagPrompt prompt for the model
	FlagPrompt = flag.String("prompt", "", "prompt for the model")
	// FlagBuild build the bin file
	FlagBuild = flag.Bool("build", false, "build the bin file")
	// FlagCompress compress the bin file
	FlagCompress = flag.Bool("compress", false, "compress the bin file")
	// FlagMach1 mach 1 mode
	FlagMach1 = flag.Bool("mach1", false, "mach 1 model")
	// FlagMach2 mach 2 model
	FlagMach2 = flag.Bool("mach2", false, "mach 2 model")
	// FlagMach3 mach 3 model
	FlagMach3 = flag.Bool("mach3", false, "mach 3 model")
	// FlagMach4 mach 4 model
	FlagMach4 = flag.Bool("mach4", false, "mach 4 model")
)

func dot(a *[InputSize]float32, b []float32) float64 {
	sum := 0.0
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return sum
}

func main() {
	flag.Parse()

	if *FlagMach1 {
		Mach1()
		return
	}

	if *FlagMach2 {
		Mach2()
		return
	}

	if *FlagMach3 {
		Mach3()
		return
	}

	if *FlagMach4 {
		Mach4()
		return
	}

	file, err := Data.Open("books/100.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	forward, reverse, code := make(map[rune]byte), make(map[byte]rune), byte(0)
	for _, v := range string(data) {
		if _, ok := forward[v]; !ok {
			forward[v] = code
			reverse[code] = v
			code++
			if code > 255 {
				panic("not enough codes")
			}
		}
	}
	length := len(forward)
	size := length

	if *FlagBuild {
		const fileName = "statistics.bin"
		_, err := os.Stat(fileName)
		counts := make([]float64, length)
		avg := make([][]float64, length)
		for i := range avg {
			avg[i] = make([]float64, size)
		}
		cov := make([][][]float64, length)
		for i := range cov {
			cov[i] = make([][]float64, size)
			for ii := range cov[i] {
				cov[i][ii] = make([]float64, size)
			}
		}
		if errors.Is(err, os.ErrNotExist) {
			out, err := os.Create(fileName)
			if err != nil {
				panic(err)
			}
			defer out.Close()

			m := mat64.NewMixer(size)
			m.Add(0)
			symbols := []rune(string(data))
			for _, symbol := range symbols {
				vector, code := m.Mix(), forward[symbol]
				counts[code]++
				for i, value := range vector {
					avg[code][i] += value
				}
				m.Add(code)
			}
			for i := range avg {
				c := counts[i]
				if c == 0 {
					continue
				}
				for ii := range avg[i] {
					avg[i][ii] /= c
				}
			}

			m = mat64.NewMixer(size)
			m.Add(0)
			cov := make([][][]float64, length)
			for i := range cov {
				cov[i] = make([][]float64, size)
				for ii := range cov[i] {
					cov[i][ii] = make([]float64, size)
				}
			}
			for _, symbol := range symbols {
				vector, code := m.Mix(), forward[symbol]
				for i, a := range vector {
					diff1 := avg[code][i] - a
					for ii, b := range vector {
						diff2 := avg[code][ii] - b
						cov[code][i][ii] += diff1 * diff2
					}
				}
				m.Add(code)
			}
			for i := range cov {
				c := counts[i]
				if c == 0 {
					continue
				}
				for ii := range cov[i] {
					for iii := range cov[i][ii] {
						cov[i][ii][iii] = cov[i][ii][ii] / c
					}
				}
			}

			buffer64 := make([]byte, 8)
			for i := range avg {
				for ii := range avg[i] {
					bits := math.Float64bits(avg[i][ii])
					for i := range buffer64 {
						buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
					}
					n, err := out.Write(buffer64)
					if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic("8 bytes should be been written")
					}
				}
			}
			for i := range cov {
				for ii := range cov[i] {
					for iii := range cov[i][ii] {
						bits := math.Float64bits(cov[i][ii][iii])
						for i := range buffer64 {
							buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
						}
						n, err := out.Write(buffer64)
						if err != nil {
							panic(err)
						}
						if n != len(buffer64) {
							panic("8 bytes should be been written")
						}
					}
				}
			}
		} else {
			input, err := os.Open(fileName)
			if err != nil {
				panic(err)
			}
			defer input.Close()

			buffer64 := make([]byte, 8)
			for i := range avg {
				for ii := range avg[i] {
					n, err := input.Read(buffer64)
					if err == io.EOF {
						panic(err)
					} else if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic(fmt.Errorf("not all bytes read: %d", n))
					}
					value := uint64(0)
					for k := 0; k < 8; k++ {
						value <<= 8
						value |= uint64(buffer64[7-k])
					}
					avg[i][ii] = math.Float64frombits(value)
				}
			}
			for i := range cov {
				for ii := range cov[i] {
					for iii := range cov[i][ii] {
						n, err := input.Read(buffer64)
						if err == io.EOF {
							panic(err)
						} else if err != nil {
							panic(err)
						}
						if n != len(buffer64) {
							panic(fmt.Errorf("not all bytes read: %d", n))
						}
						value := uint64(0)
						for k := 0; k < 8; k++ {
							value <<= 8
							value |= uint64(buffer64[7-k])
						}
						cov[i][ii][iii] = math.Float64frombits(value)
					}
				}
			}
		}

		out, err := os.Create("model.bin")
		if err != nil {
			panic(err)
		}
		defer out.Close()

		{
			buffer64 := make([]byte, 8)
			for i := range avg {
				for ii := range avg[i] {
					bits := math.Float64bits(avg[i][ii])
					for i := range buffer64 {
						buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
					}
					n, err := out.Write(buffer64)
					if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic("8 bytes should be been written")
					}
				}
			}
		}

		rng := rand.New(rand.NewSource(1))
		for s := range avg {
			set := tf64.NewSet()
			set.Add("A", size, size)
			set.Add("AI", size, size)

			for i := range set.Weights {
				w := set.Weights[i]
				if strings.HasPrefix(w.N, "b") {
					w.X = w.X[:cap(w.X)]
					w.States = make([][]float64, StateTotal)
					for ii := range w.States {
						w.States[ii] = make([]float64, len(w.X))
					}
					continue
				}
				factor := math.Sqrt(2.0 / float64(w.S[0]))
				for range cap(w.X) {
					w.X = append(w.X, rng.NormFloat64()*factor)
				}
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
			}

			others := tf64.NewSet()
			others.Add("E", size, size)
			others.Add("I", size, size)
			E := others.ByName["E"]
			for i := range cov {
				for ii := range cov[i] {
					E.X = append(E.X, cov[s][i][ii])
				}
			}
			I := others.ByName["I"]
			for i := range size {
				for ii := range size {
					if i == ii {
						I.X = append(I.X, 1)
					} else {
						I.X = append(I.X, 0)
					}
				}
			}

			{
				loss := tf64.Sum(tf64.Quadratic(others.Get("E"), tf64.Mul(set.Get("A"), set.Get("A"))))

				points := make(plotter.XYs, 0, 8)
				for i := range 1024 {
					pow := func(x float64) float64 {
						y := math.Pow(x, float64(i+1))
						if math.IsNaN(y) || math.IsInf(y, 0) {
							return 0
						}
						return y
					}

					set.Zero()
					others.Zero()
					cost := tf64.Gradient(loss).X[0]
					if math.IsNaN(cost) || math.IsInf(cost, 0) {
						fmt.Println(i, cost)
						break
					}

					norm := 0.0
					for _, p := range set.Weights {
						for _, d := range p.D {
							norm += d * d
						}
					}
					norm = math.Sqrt(norm)
					b1, b2 := pow(B1), pow(B2)
					scaling := 1.0
					if norm > 1 {
						scaling = 1 / norm
					}
					for _, w := range set.Weights {
						if w.N != "A" {
							continue
						}
						for ii, d := range w.D {
							g := d * scaling
							m := B1*w.States[StateM][ii] + (1-B1)*g
							v := B2*w.States[StateV][ii] + (1-B2)*g*g
							w.States[StateM][ii] = m
							w.States[StateV][ii] = v
							mhat := m / (1 - b1)
							vhat := v / (1 - b2)
							if vhat < 0 {
								vhat = 0
							}
							w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
						}
					}
					points = append(points, plotter.XY{X: float64(i), Y: cost})
				}

				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("plots/%d_epochs.png", s))
				if err != nil {
					panic(err)
				}
			}
			{
				buffer64 := make([]byte, 8)
				a := set.ByName["A"]
				for i := range a.X {
					bits := math.Float64bits(a.X[i])
					for i := range buffer64 {
						buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
					}
					n, err := out.Write(buffer64)
					if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic("8 bytes should be been written")
					}
				}
			}

			{
				const Eta = 1.0e-1
				loss := tf64.Sum(tf64.Quadratic(others.Get("I"), tf64.Mul(set.Get("A"), set.Get("AI"))))

				points := make(plotter.XYs, 0, 8)
				for i := range 128 * 1024 {
					pow := func(x float64) float64 {
						y := math.Pow(x, float64(i+1))
						if math.IsNaN(y) || math.IsInf(y, 0) {
							return 0
						}
						return y
					}

					set.Zero()
					others.Zero()
					cost := tf64.Gradient(loss).X[0]
					if math.IsNaN(cost) || math.IsInf(cost, 0) {
						fmt.Println(i, cost)
						break
					}

					norm := 0.0
					for _, p := range set.Weights {
						if p.N != "AI" {
							continue
						}
						for _, d := range p.D {
							norm += d * d
						}
					}
					norm = math.Sqrt(norm)
					b1, b2 := pow(B1), pow(B2)
					scaling := 1.0
					if norm > 1 {
						scaling = 1 / norm
					}
					for _, w := range set.Weights {
						if w.N != "AI" {
							continue
						}
						for ii, d := range w.D {
							g := d * scaling
							m := B1*w.States[StateM][ii] + (1-B1)*g
							v := B2*w.States[StateV][ii] + (1-B2)*g*g
							w.States[StateM][ii] = m
							w.States[StateV][ii] = v
							mhat := m / (1 - b1)
							vhat := v / (1 - b2)
							if vhat < 0 {
								vhat = 0
							}
							w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
						}
					}
					points = append(points, plotter.XY{X: float64(i), Y: cost})
				}

				p := plot.New()

				p.Title.Text = "epochs vs cost"
				p.X.Label.Text = "epochs"
				p.Y.Label.Text = "cost"

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(1)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(scatter)

				err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("plots/%d_inverse_epochs.png", s))
				if err != nil {
					panic(err)
				}
			}
			{
				buffer64 := make([]byte, 8)
				ai := set.ByName["AI"]
				for i := range ai.X {
					bits := math.Float64bits(ai.X[i])
					for i := range buffer64 {
						buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
					}
					n, err := out.Write(buffer64)
					if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic("8 bytes should be been written")
					}
				}
			}
		}
		return
	}

	input, err := os.Open("model.bin")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	avg, a, ai := make([]mat64.Matrix, length), make([]mat64.Matrix, length), make([]mat64.Matrix, length)
	for i := range length {
		avg[i] = mat64.NewMatrix(size, 1)
		a[i] = mat64.NewMatrix(size, size)
		ai[i] = mat64.NewMatrix(size, size)
	}
	{
		buffer64 := make([]byte, 8)
		for i := range avg {
			for range avg[i].Rows {
				for range avg[i].Cols {
					n, err := input.Read(buffer64)
					if err == io.EOF {
						panic(err)
					} else if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic(fmt.Errorf("not all bytes read: %d", n))
					}
					value := uint64(0)
					for k := 0; k < 8; k++ {
						value <<= 8
						value |= uint64(buffer64[7-k])
					}
					avg[i].Data = append(avg[i].Data, math.Float64frombits(value))
				}
			}
		}
	}
	{
		buffer64 := make([]byte, 8)
		for i := range a {
			for range a[i].Rows {
				for range a[i].Cols {
					n, err := input.Read(buffer64)
					if err == io.EOF {
						panic(err)
					} else if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic(fmt.Errorf("not all bytes read: %d", n))
					}
					value := uint64(0)
					for k := 0; k < 8; k++ {
						value <<= 8
						value |= uint64(buffer64[7-k])
					}
					a[i].Data = append(a[i].Data, math.Float64frombits(value))
				}
			}
			for range ai[i].Rows {
				for range ai[i].Cols {
					n, err := input.Read(buffer64)
					if err == io.EOF {
						panic(err)
					} else if err != nil {
						panic(err)
					}
					if n != len(buffer64) {
						panic(fmt.Errorf("not all bytes read: %d", n))
					}
					value := uint64(0)
					for k := 0; k < 8; k++ {
						value <<= 8
						value |= uint64(buffer64[7-k])
					}
					ai[i].Data = append(ai[i].Data, math.Float64frombits(value))
				}
			}
		}
		_, err := input.Read(buffer64)
		if err != io.EOF {
			panic("not at the end")
		}
	}
	count, total := 0.0, 0.0
	for i := range a {
		x := a[i].MulT(ai[i])
		for ii := range x.Rows {
			for iii := range x.Cols {
				if ii == iii {
					value := x.Data[ii*x.Cols+iii]
					if value < .9 {
						count++
					}
					total++
				}
			}
		}
	}
	fmt.Println(count, total, count/total)
}
