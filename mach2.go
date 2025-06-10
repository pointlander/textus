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
)

// Mach2 mach 2 model
func Mach2() {
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

		rng := rand.New(rand.NewSource(1))
		current := m.Mix()
		var search func(samples, begin, end int) byte
		search = func(samples, begin, end int) byte {
			buffer, vector := [ItemSize]byte{}, [InputSize]float32{}
			if end-begin <= Samples {
				input.Seek(int64(begin*len(buffer)), 0)
				max, symbol := float32(0.0), byte(0)
				for range end - begin {
					n, err := input.Read(buffer[:])
					if err == io.EOF {
						panic("symbol not found")
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
					if a := CS(vector[:], current[:]); a > max {
						max, symbol = a, buffer[len(buffer)-1]
					}
				}
				return symbol
			}
			a, b := float32(0.0), float32(0.0)
			aa, bb := make([]float32, 0, Samples), make([]float32, 0, Samples)
			for range Samples {
				index := rng.Intn(end-begin)/2 + begin
				input.Seek(int64(index*len(buffer)), 0)
				n, err := input.Read(buffer[:])
				if err == io.EOF {
					continue
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
				cs := CS(vector[:], current[:])
				a += cs
				aa = append(aa, cs)
			}
			for range Samples {
				index := end - rng.Intn(end-begin)/2
				input.Seek(int64(index*len(buffer)), 0)
				n, err := input.Read(buffer[:])
				if err == io.EOF {
					continue
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
				cs := CS(vector[:], current[:])
				b += cs
				bb = append(bb, cs)
			}
			a /= float32(Samples)
			b /= float32(Samples)
			va, vb := float32(0.0), float32(0.0)
			for _, v := range aa {
				diff := v - a
				va += diff * diff
			}
			va /= float32(Samples)
			for _, v := range bb {
				diff := v - b
				vb += diff * diff
			}
			vb /= float32(Samples)
			if samples > 256 {
				samples >>= 1
			}
			if va < vb {
				return search(samples, begin, begin+(end-begin)/2)
			}
			return search(samples, begin+(end-begin)/2, end)
		}
		for range 256 {
			symbol := search(Samples, 0, int(length))
			fmt.Printf("%c", reverse[symbol])
			m.Add(symbol)
			current = m.Mix()
		}
		return
	}

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
}
