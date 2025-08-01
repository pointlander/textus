// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"sort"
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
	// FlagMach5 mach 5 model
	FlagMach5 = flag.Bool("mach5", false, "mach 5 model")
)

func dot(a *[InputSize]float32, b []float32) float64 {
	sum := 0.0
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return sum
}

// L2 is the L2 norm
func L2(a, b []float64) float64 {
	c := 0.0
	for i, v := range a {
		diff := v - b[i]
		c += diff * diff
	}
	return c
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

	if *FlagMach5 {
		Mach5()
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

	type Context [2]byte
	type Vector struct {
		Vector []float32
		Symbol byte
	}
	if *FlagBuild {
		//model := make(map[Context][]Vector)
		m := NewBasic(256)
		m.Add(0)
		buffer32, buffer8 := make([]byte, 4), make([]byte, 1)
		for _, v := range string(data) {
			vector := m.Mix()
			context := Context{m.Markov[0], m.Markov[1]}
			err := os.MkdirAll(path.Join("model", fmt.Sprintf("%d", context[0])), 0750)
			if err != nil {
				panic(err)
			}
			name := path.Join("model", fmt.Sprintf("%d", context[0]), fmt.Sprintf("%d", context[1]))
			output, err := os.OpenFile(name, os.O_RDWR, 0750)
			if err != nil {
				output, err = os.Create(name)
				if err != nil {
					panic(err)
				}
			}
			_, err = output.Seek(0, 2)
			if err != nil {
				panic(err)
			}
			for _, v := range vector {
				bits := math.Float32bits(v)
				for i := range buffer32 {
					buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := output.Write(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should be been written")
				}
			}
			buffer8[0] = forward[v]
			n, err := output.Write(buffer8)
			if err != nil {
				panic(err)
			}
			if n != 1 {
				panic("1 byte should be been written")
			}
			/*x := model[context]
			x = append(x, Vector{
				Vector: vector,
				Symbol: forward[v],
			})
			model[context] = x*/
			output.Close()
			m.Add(forward[v])
		}
		return
	}

	if *FlagPrompt != "" {
		rng := rand.New(rand.NewSource(1))
		type Sample struct {
			Sample      string
			Probability float32
		}
		samples := []Sample{}
		fmt.Println(*FlagPrompt)
		for range 33 {
			m := NewBasic(256)
			txt := []rune(*FlagPrompt)
			for _, v := range txt {
				m.Add(forward[v])
			}
			sample := Sample{}
			for range 33 {
				current := m.Mix()
				context := Context{m.Markov[0], m.Markov[1]}
				//max, symbol := float32(0.0), byte(0)
				symbol := byte(0)
				histogram, count := make([]float32, length), float32(0.0)
				name := path.Join("model", fmt.Sprintf("%d", context[0]), fmt.Sprintf("%d", context[1]))
				input, err := os.Open(name)
				if err == nil {
					buffer, vector := [ItemSize]byte{}, [InputSize]float32{}
					for {
						n, err := input.Read(buffer[:])
						if err == io.EOF {
							err := input.Close()
							if err != nil {
								panic(err)
							}
							break
						} else if err != nil {
							panic(err)
						}
						if n != len(buffer) {
							panic("not all bytes read")
						}
						for j := range vector {
							value := uint32(0)
							for k := 0; k < 4; k++ {
								value <<= 8
								value |= uint32(buffer[j*4+3-k])
							}
							vector[j] = math.Float32frombits(value)
						}
						a, s := CS(vector[:], current[:]), buffer[len(buffer)-1]
						/*if a > max {
							max, symbol = a, s
						}*/
						histogram[s] += a
						count += a
					}
				} else {
					for i := range length {
						name := path.Join("model", fmt.Sprintf("%d", context[0]), fmt.Sprintf("%d", i))
						input, err := os.Open(name)
						if err == nil {
							buffer, vector := [ItemSize]byte{}, [InputSize]float32{}
							for {
								n, err := input.Read(buffer[:])
								if err == io.EOF {
									err := input.Close()
									if err != nil {
										panic(err)
									}
									break
								} else if err != nil {
									panic(err)
								}
								if n != len(buffer) {
									panic("not all bytes read")
								}
								for j := range vector {
									value := uint32(0)
									for k := 0; k < 4; k++ {
										value <<= 8
										value |= uint32(buffer[j*4+3-k])
									}
									vector[j] = math.Float32frombits(value)
								}
								a, s := CS(vector[:], current[:]), buffer[len(buffer)-1]
								/*if a > max {
									max, symbol = a, s
								}*/
								histogram[s] += a
								count += a
							}
						}
					}
				}
				for i, c := range histogram {
					histogram[i] = c / count
				}
				sum, selected := float32(0.0), rng.Float32()
				for i, c := range histogram {
					sum += c
					if selected < sum {
						symbol = byte(i)
						sample.Sample += fmt.Sprintf("%c", reverse[byte(i)])
						sample.Probability += c
						break
					}
				}
				//fmt.Printf("%c %d\n", reverse[symbol], reverse[symbol])
				//txt = append(txt, reverse[symbol])
				m.Add(symbol)
			}
			samples = append(samples, sample)
		}

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Probability < samples[j].Probability
		})
		for _, sample := range samples {
			fmt.Println("_______________________________________________________________")
			fmt.Println(sample.Probability)
			fmt.Println(sample.Sample)
		}
		return
	}
}
