// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"flag"
	"io"
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

func dot(a, b *[InputSize]float32) float64 {
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
	size := len(forward)

	if *FlagBuild {
		m := NewFiltered()
		m.Add(0)
		avg := make([][]float32, size)
		for i := range avg {
			avg[i] = make([]float32, 256)
		}
		symbols := []rune(string(data))
		for _, symbol := range symbols {
			vector, code := m.Mix(), forward[symbol]
			for i, value := range vector {
				avg[code][i] += value
			}
			m.Add(code)
		}
		for i := range avg {
			for ii := range avg[i] {
				avg[i][ii] /= float32(len(symbols))
			}
		}

		m = NewFiltered()
		m.Add(0)
		cov := make([][][]float32, size)
		for i := range cov {
			cov[i] = make([][]float32, 256)
			for ii := range cov[i] {
				cov[i][ii] = make([]float32, 256)
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
			for ii := range cov[i] {
				for iii := range cov[i][ii] {
					cov[i][ii][iii] = cov[i][ii][ii] / float32(len(symbols))
				}
			}
		}
	}
}
