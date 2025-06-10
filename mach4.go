// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"

	"github.com/alixaxel/pagerank"
)

// Mach4 mach 4 model
func Mach4() {
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

	if *FlagPrompt != "" {
		m := NewFiltered()
		for _, v := range []rune(*FlagPrompt) {
			m.Add(forward[v])
		}

		input, err := os.Open("db.bin")
		if err != nil {
			panic(err)
		}
		defer input.Close()

		info, err := input.Stat()
		if err != nil {
			panic(err)
		}
		length := info.Size() / ItemSize

		type Item struct {
			Vector [InputSize]float32
			Symbol byte
		}
		items := make([]Item, length)
		buffer, vec := [ItemSize]byte{}, [InputSize]float32{}
		for x := range length {
			n, err := input.Read(buffer[:])
			if err == io.EOF {
				panic("symbol not found")
			} else if err != nil {
				panic(err)
			}
			if n != len(buffer) {
				panic("not all bytes read")
			}
			for j := range vec {
				value := uint32(0)
				for k := 0; k < 4; k++ {
					value <<= 8
					value |= uint32(buffer[j*4+3-k])
				}
				vec[j] = math.Float32frombits(value)
			}
			items[x].Vector = vec
			items[x].Symbol = buffer[ItemSize-1]
		}

		if *FlagCompress {
			test, err := os.Create("test.txt")
			if err != nil {
				panic(err)
			}
			defer test.Close()

			for i := 0; i < int(length)-512; i += 512 {
				histogram := [256]uint64{}
				begin, end := i, i+1024
				if end > len(items) {
					end = len(items)
				}
				items := items[begin:end]
				for j := range items {
					histogram[items[j].Symbol]++
				}
				fmt.Fprintln(test, histogram)
			}
			return
		}

		cpus := runtime.NumCPU()
		count := len(items) / cpus
		type Result struct {
			Max    float64
			Symbol byte
			Vector [InputSize]float32
			Rank   float64
		}

		var search func(rng *rand.Rand, current [InputSize]float32) (float64, byte)
		search = func(rng *rand.Rand, current [InputSize]float32) (float64, byte) {
			results := make(chan [10]Result, 8)
			for i := range cpus {
				begin, end := i*count, (i+1)*count
				if end > len(items) {
					end = len(items)
				}
				go func(begin, end int) {
					items := items[begin:end]
					var result [10]Result
					for x := range items {
						j, a := 0, dot(&items[x].Vector, &current)
						for j < len(result) && a > result[j].Max {
							if j > 0 {
								result[j-1] = result[j]
							}
							j++
						}
						if j > 0 {
							result[j-1] = Result{a, items[x].Symbol, items[x].Vector, 0.0}
						}
					}
					results <- result
				}(begin, end)
			}

			max, symbol := 0.0, byte(0)
			combine := make([]Result, 0, 8)
			for range cpus {
				result := <-results
				combine = append(combine, result[:]...)
			}
			sort.Slice(combine, func(i, j int) bool {
				return combine[i].Max > combine[j].Max
			})
			graph := pagerank.NewGraph()
			for i := 0; i < len(combine); i++ {
				for j := 0; j < len(combine); j++ {
					p := dot(&combine[i].Vector, &combine[j].Vector)
					graph.Link(uint32(i), uint32(j), p)
				}
			}
			graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
				combine[node].Rank = rank
			})
			total, selection, index := 0.0, rng.Float64(), 0
			for j := range combine {
				total += combine[j].Rank
				if selection < total {
					index = j
					break
				}
			}
			max, symbol = combine[index].Max, combine[index].Symbol
			return max, symbol
		}
		rng := rand.New(rand.NewSource(1))
		max, symbols := 0.0, ""
		for i := 0; i < 8; i++ {
			sum, s := 0.0, ""
			cp := m.Copy()
			for range 256 {
				current := cp.Mix()
				max, symbol := search(rng, current)
				sum += max
				s += fmt.Sprintf("%c", reverse[symbol])
				cp.Add(symbol)
			}
			if sum > max {
				max, symbols = sum, s
			}
		}
		fmt.Println(symbols)
		return
	}

	if *FlagBuild {
		db, err := os.Create("db.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()

		m := NewFiltered()
		m.Add(0)
		buffer32, buffer8 := make([]byte, 4), make([]byte, 1)
		for _, v := range string(data) {
			vector := m.Mix()
			for _, v := range vector {
				bits := math.Float32bits(v)
				for i := range buffer32 {
					buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := db.Write(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should be been written")
				}
			}
			buffer8[0] = forward[v]
			n, err := db.Write(buffer8)
			if err != nil {
				panic(err)
			}
			if n != 1 {
				panic("1 byte should be been written")
			}
			m.Add(forward[v])
		}
		return
	}
}
