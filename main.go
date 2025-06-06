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
	"math/bits"
	"math/rand"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/textus/vector"

	"github.com/alixaxel/pagerank"
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
