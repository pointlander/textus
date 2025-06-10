// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/bits"
	"math/rand"
	"os"

	"github.com/pointlander/textus/vector"
)

// Mach3 mach 3 model
func Mach3() {
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

	rng := rand.New(rand.NewSource(1))
	var vectors [128][2][256]float32
	for i := range vectors {
		for j := range vectors[i] {
			v := vectors[i][j][:]
			for k := range v {
				v[k] = rng.Float32()
			}
			vv := sqrt(vector.Dot(v, v))
			for k := range v {
				v[k] /= vv
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
		length := info.Size() / (2 * 8)

		buffer64 := make([]byte, 8)
		items := make([][2]uint64, length)
		for i := range items {
			for j := range items[i] {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					panic("symbol not found")
				} else if err != nil {
					panic(err)
				}
				if n != len(buffer64) {
					panic("not all bytes read")
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				items[i][j] = value
			}
		}

		datum := []rune(string(data))
		for i := 0; i < 256; i++ {
			vec := m.Mix()
			bit := [2]uint64{}
			for j := range vectors {
				bit[j/64] <<= 1
				a, b := vector.Dot(vectors[j][0][:], vec[:]), vector.Dot(vectors[j][1][:], vec[:])
				if a > b {
					bit[j/64] |= 1
				}
			}

			min, index := 64, 0
			for j := range items {
				if a := bits.OnesCount64(items[j][0]^bit[0]) + bits.OnesCount64(items[j][1]^bit[1]); a < min {
					min, index = a, j
				}
			}
			symbol := datum[index]
			fmt.Printf("%c", symbol)
			m.Add(forward[symbol])
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
	buffer64 := make([]byte, 8)
	for _, v := range string(data) {
		vec := m.Mix()
		bits := [2]uint64{}
		for j := range bits {
			for i := range vectors {
				bits[j] <<= 1
				a, b := vector.Dot(vectors[i][0][:], vec[:]), vector.Dot(vectors[i][1][:], vec[:])
				if a > b {
					bits[j] |= 1
				}
			}

			for i := range buffer64 {
				buffer64[i] = byte((bits[j] >> (8 * i)) & 0xFF)
			}
			n, err := db.Write(buffer64)
			if err != nil {
				panic(err)
			}
			if n != len(buffer64) {
				panic("8 bytes should be been written")
			}
		}
		m.Add(forward[v])
	}
}
