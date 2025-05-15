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
	"os"
)

const (
	// VectorSize is the size of a vector
	VectorSize = InputSize * 4
	// ItemSize is the size of a row
	ItemSize = VectorSize + 1
)

//go:embed books/*
var Data embed.FS

var (
	// FlagPrompt prompt for the model
	FlagPrompt = flag.String("prompt", "", "prompt for the model")
)

func main() {
	flag.Parse()

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

	if *FlagPrompt != "" {
		m := NewFiltered()
		for _, v := range []byte(*FlagPrompt) {
			m.Add(v)
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
		length := info.Size()
		items := length / ItemSize
		buffer, vector := [ItemSize]byte{}, [InputSize]float32{}
		for i := 0; i < 33; i++ {
			current := m.Mix()
			max, symbol := float32(0.0), byte(0)
			for range items {
				n, err := input.Read(buffer[:])
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
			input.Seek(0, 0)
		}
		fmt.Println()
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
	for _, v := range data {
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
		buffer8[0] = v
		n, err := db.Write(buffer8)
		if err != nil {
			panic(err)
		}
		if n != 1 {
			panic("1 byte should be been written")
		}
		m.Add(v)
	}
}
