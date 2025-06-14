// Copyright 2025 The Textus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"fmt"
	"math"
)

const (
	// InputSize is the size of the input
	InputSize = 256
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e38*math.SmallestNonzeroFloat32
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i, value := range a {
		sum += value * b[i]
	}
	return sum
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float64 matrices
func (m Matrix) Add(n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix) Softmax(T float64) Matrix {
	output := NewMatrix(m.Cols, m.Rows)
	max := 0.0
	for _, v := range m.Data {
		v /= T
		if v > max {
			max = v
		}
	}
	s := max * S
	for i := 0; i < len(m.Data); i += m.Cols {
		sum := 0.0
		values := make([]float64, m.Cols)
		for j, value := range m.Data[i : i+m.Cols] {
			values[j] = math.Exp(value/T - s)
			sum += values[j]
		}
		for _, value := range values {
			output.Data = append(output.Data, value/sum)
		}
	}
	return output
}

// Entropy calculates the entropy of the matrix rows
func (m Matrix) Entropy() Matrix {
	output := NewMatrix(m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := 0.0
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * math.Log(value)
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}

// T tramsposes a matrix
func (m Matrix) T() Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// AddRow adds a row to a matrix
func (m Matrix) AddRow(row []float64) Matrix {
	if len(row) != m.Cols {
		panic("incorrect number of columns")
	}
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows + 1,
		Data: make([]float64, m.Cols*m.Rows),
	}
	copy(o.Data, m.Data)
	o.Data = append(o.Data, row...)
	return o
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(input Matrix) []float64 {
	values := make([]float64, input.Rows)
	V := input.T()
	output := make([]float64, input.Cols)
	for i := 0; i < input.Rows; i++ {
		K := input.Data[i*input.Cols : (i+1)*input.Cols]
		for j := 0; j < input.Rows; j++ {
			Q := input.Data[j*input.Cols : (j+1)*input.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			output[j] += dot(values, V)
		}
	}
	aa := math.Sqrt(dot(output[:], output[:]))
	for i, v := range output {
		output[i] = v / aa
	}
	return output
}

// CS is float64 normalized cosine similarity
func CS(a []float64, b []float64) float64 {
	aa, bb, ab := dot(a, a), dot(b, b), dot(a, b)
	if aa <= 0 {
		return 0
	}
	if bb <= 0 {
		return 0
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}
