// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
)

// Mach1 model
func Mach1() {
	if *FlagPrompt != "" {
		m := NewFiltered()
		for _, v := range []byte(*FlagPrompt) {
			m.Add(v)
		}
		input := make([]*os.File, 5)
		for i := range input {
			var err error
			input[i], err = os.Open(fmt.Sprintf("db.bin.%d", i))
			if err != nil {
				panic(err)
			}
			defer input[i].Close()
		}
		items := make([]int64, 5)
		for i := range items {
			info, err := input[i].Stat()
			if err != nil {
				panic(err)
			}
			length := info.Size()
			if i == 0 {
				items[i] = length / ItemSize
			} else {
				items[i] = length / VectorSize
			}
		}
		buffer, vectorBuffer, vector := [ItemSize]byte{}, [VectorSize]byte{}, [InputSize]float32{}
		for i := 0; i < 128; i++ {
			current := m.Mix()
			max, index := float32(0.0), 0
			for j := range items[4] {
				n, err := input[4].Read(vectorBuffer[:])
				if err == io.EOF {
					break
				} else if err != nil {
					panic(err)
				}
				if n != len(vectorBuffer) {
					panic("not all bytes read")
				}
				for j := range vector {
					value := uint32(0)
					for k := 0; k < 4; k++ {
						value <<= 8
						value |= uint32(vectorBuffer[j*4+3-k])
					}
					vector[j] = math.Float32frombits(value)
				}
				if a := CS(vector[:], current[:]); a > max {
					max, index = a, int(j)
				}
			}
			input[4].Seek(0, 0)
			for j := 3; j > 0; j-- {
				input[j].Seek(int64(index*8*len(vectorBuffer)), 0)
				max, index = float32(0.0), 0
				for k := index * 8; k < (index+1)*8; k++ {
					n, err := input[j].Read(vectorBuffer[:])
					if err == io.EOF {
						break
					} else if err != nil {
						panic(err)
					}
					if n != len(vectorBuffer) {
						panic(fmt.Errorf("not all bytes read: %d", n))
					}
					for j := range vector {
						value := uint32(0)
						for k := 0; k < 4; k++ {
							value <<= 8
							value |= uint32(vectorBuffer[j*4+3-k])
						}
						vector[j] = math.Float32frombits(value)
					}
					if a := CS(vector[:], current[:]); a > max {
						max, index = a, int(k)
					}
				}
			}
			max, symbol := float32(0.0), byte(0)
			input[0].Seek(int64(index*8*len(buffer)), 0)
			for k := index * 8; k < (index+1)*8; k++ {
				n, err := input[0].Read(buffer[:])
				if err == io.EOF {
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
				if a := CS(vector[:], current[:]); a > max {
					max, symbol = a, buffer[ItemSize-1]
				}
			}
			fmt.Printf("%c", symbol)
			m.Add(symbol)
		}
		fmt.Println()
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

	length := bits.Len64(uint64(len(data)))
	length /= 4
	db := make([]*os.File, length)
	vectors := make([][InputSize]float32, length)
	eight := make([]int, length)
	fmt.Println(length)
	for i := range db {
		var err error
		db[i], err = os.Create(fmt.Sprintf("db.bin.%d", i))
		if err != nil {
			panic(err)
		}
		defer db[i].Close()
	}

	m := NewFiltered()
	m.Add(0)
	buffer32, buffer8 := make([]byte, 4), make([]byte, 1)
	for _, v := range data {
		vector := m.Mix()
		for _, v := range vector {
			bits := math.Float32bits(v)
			for i := range buffer32 {
				buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := db[0].Write(buffer32)
			if err != nil {
				panic(err)
			}
			if n != len(buffer32) {
				panic("4 bytes should be been written")
			}
		}

		for i, v := range vector {
			vectors[0][i] += v
		}
		eight[0]++
		index := 1
		for index < length && eight[index-1]%8 == 0 && eight[index-1] != 0 {
			for j, v := range vectors[index-1] {
				bits := math.Float32bits(v)
				for i := range buffer32 {
					buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := db[index].Write(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should be been written")
				}
				vectors[index][j] += v
				vectors[index-1][j] = 0.0
			}
			eight[index]++
			eight[index-1] = 0
			index++
		}

		buffer8[0] = v
		n, err := db[0].Write(buffer8)
		if err != nil {
			panic(err)
		}
		if n != 1 {
			panic("1 byte should be been written")
		}
		m.Add(v)
	}
}
