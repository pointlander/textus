// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"embed"
	"flag"
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
}
