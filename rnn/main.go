// Copyright 2020 The Models Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

const (
	// NumberOfVerses is the number of verses in the bible
	NumberOfVerses = 31102
	// Symbols is the number of symbols
	Symbols = 256
	// Space is the state space of the Println
	Space = 256
	// Width is the width of the neural network
	Width = Symbols + Space
	// Batch is the batch size
	Batch = 256
	// Scale scales the neural network
	Scale = 2
)

var (
	// Nets the number of nets to run in parallel
	Nets = runtime.NumCPU()
	// PatternBook marks the start of a book
	PatternBook = regexp.MustCompile(`\r\n\r\n\r\n\r\n[A-Za-z]+([ \t]+[A-Za-z:]+)*\r\n\r\n`)
	// PatternVerse is a verse
	PatternVerse = regexp.MustCompile(`\d+[:]\d+[A-Za-z:.,?;"' ()\t\r\n]+`)
	// PatternWord is for splitting into words
	PatternWord = regexp.MustCompile(`[ \t\r\n]+`)
	// FlagVerbose enables verbose mode
	FlagVerbose = flag.Bool("verbose", false, "verbose mode")
	// FlagLearn learn the model
	FlagLearn = flag.String("learn", "", "learning mode")
	// FlagInference load weights and generate probable strings
	FlagInference = flag.String("inference", "", "inference mode")
)

// Testament is a bible testament
type Testament struct {
	Name  string
	Books []Book
}

// Book is a book of the bible
type Book struct {
	Name   string
	Verses []Verse
}

// Verse is a bible verse
type Verse struct {
	Number string
	Verse  string
}

func main() {
	flag.Parse()

	if *FlagLearn != "" {
		switch *FlagLearn {
		case "fixed":
			FixedLearn()
		case "variable":
			VariableLearn()
		case "hierarchical":
			HierarchicalLearn()
		}

		return
	} else if *FlagInference != "" {
		Inference()
		return
	} else {
		verses, words, max := Verses()
		maxWord := 0
		for _, word := range words {
			if length := len(word); length > maxWord {
				maxWord = length
			}
		}
		fmt.Printf("number of verses %d\n", len(verses))
		fmt.Printf("max verse length %d\n", max)
		fmt.Printf("number of unique words %d\n", len(words))
		fmt.Printf("max word length %d\n", maxWord)
		return
	}
}

// Inference inference mode
func Inference() {
	in, err := ioutil.ReadFile(*FlagInference)
	if err != nil {
		panic(err)
	}
	set := Set{}
	err = proto.Unmarshal(in, &set)
	if err != nil {
		panic(err)
	}
	w1, b1 := tf32.NewV(2*Width, Scale*2*Width), tf32.NewV(Scale*2*Width)
	w2, b2 := tf32.NewV(Scale*4*Width, Width), tf32.NewV(Width)
	for _, weights := range set.Weights {
		switch weights.Name {
		case "w1":
			w1.X = weights.Values
		case "b1":
			b1.X = weights.Values
		case "w2":
			w2.X = weights.Values
		case "b2":
			b2.X = weights.Values
		}
	}
	bestSum, best := float32(0.0), []rune{}
	var search func(depth int, most []rune, previous *tf32.V, sum float32)
	search = func(depth int, most []rune, previous *tf32.V, sum float32) {
		if depth > 1 {
			if sum > bestSum {
				best, bestSum = most, sum
				fmt.Println(best)
				fmt.Println(string(best))
			}
			return
		}

		input, state := tf32.NewV(2*Symbols, 1), tf32.NewV(2*Space, 1)
		input.X = input.X[:cap(input.X)]
		state.X = state.X[:cap(state.X)]
		l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(input.Meta(), previous.Meta())), b1.Meta()))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
		setSymbol := func(s rune) {
			for i := range input.X {
				if i%2 == 0 {
					input.X[i] = 0
				} else {
					input.X[i] = 0
				}
			}
			symbol := 2 * int(s)
			input.X[symbol] = 0
			input.X[symbol+1] = 1
		}
		setSymbol(most[len(most)-1])
		l2(func(a *tf32.V) bool {
			symbols := a.X[:2*Symbols]
			copy(state.X, a.X[2*Symbols:])
			for i, symbol := range symbols {
				if i&1 == 1 {
					cp := make([]rune, len(most))
					copy(cp, most)
					cp = append(cp, rune(i>>1))
					search(depth+1, cp, &state, sum+symbol)
				}
			}
			return true
		})
	}
	state := tf32.NewV(2*Space, 1)
	state.X = state.X[:cap(state.X)]
	search(0, []rune{'Y'}, &state, 0)
}

// HierarchicalLearn learns the ierarchical encoder decoder rnn model for words
func HierarchicalLearn() {
	_, words, _ := Verses()
	fmt.Println(len(words))
	initial := tf32.NewV(2*Space, 1)
	initial.X = initial.X[:cap(initial.X)]
	aw1, ab1 := tf32.NewV(2*Width, Scale*2*Width), tf32.NewV(Scale*2*Width)
	aw2, ab2 := tf32.NewV(Scale*4*Width, Space), tf32.NewV(Space)
	bw1, bb1 := tf32.NewV(2*Space, Scale*2*Width), tf32.NewV(Scale*2*Width)
	bw2, bb2 := tf32.NewV(Scale*4*Width, Width), tf32.NewV(Width)
	parameters := []*tf32.V{
		&aw1, &ab1, &aw2, &ab2,
		&bw1, &bb1, &bw2, &bb2,
	}
	for _, p := range parameters {
		factor := float32(math.Sqrt(float64(p.S[0])))
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, Random32(-1, 1)/factor)
		}
	}

	deltas := make([][]float32, 0, len(parameters))
	for _, p := range parameters {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	done := make(chan float32, 8)
	learn := func(parameters []*tf32.V, word string) {
		aw1, ab1, aw2, ab2 := parameters[0], parameters[1], parameters[2], parameters[3]
		bw1, bb1, bw2, bb2 := parameters[4], parameters[5], parameters[6], parameters[7]
		wordSymbols := []rune(word)
		symbols := make([]tf32.V, 0, len(wordSymbols))
		for _, s := range wordSymbols {
			symbol := tf32.NewV(2*Symbols, 1)
			symbol.X = symbol.X[:cap(symbol.X)]
			index := 2 & int(s)
			symbol.X[index] = 0
			symbol.X[index+1] = 1
			symbols = append(symbols, symbol)
		}

		l1 := tf32.Everett(tf32.Add(tf32.Mul(aw1.Meta(), tf32.Concat(symbols[0].Meta(), initial.Meta())), ab1.Meta()))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(aw2.Meta(), l1), ab2.Meta()))
		for j := 1; j < len(symbols); j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(aw1.Meta(), tf32.Concat(symbols[j].Meta(), l2)), ab1.Meta()))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(aw2.Meta(), l1), ab2.Meta()))
		}

		l1 = tf32.Everett(tf32.Add(tf32.Mul(bw1.Meta(), l2), bb1.Meta()))
		l2 = tf32.Everett(tf32.Add(tf32.Mul(bw2.Meta(), l1), bb2.Meta()))
		cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0].Meta()))
		for j := 1; j < len(symbols); j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(bw1.Meta(), tf32.Slice(l2, space.Meta())), bb1.Meta()))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(bw2.Meta(), l1), bb2.Meta()))
			cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[j].Meta())))
		}

		done <- tf32.Gradient(cost).X[0]
	}

	iterations := 100
	alpha, eta := float32(.3), float32(.3/float64(Nets))
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range words {
			j := i + rand.Intn(len(words)-i)
			words[i], words[j] = words[j], words[i]
		}

		total := float32(0.0)
		for j := 0; j < len(words); j += Nets {
			flight, copies := 0, make([][]*tf32.V, 0, Nets)
			for k := 0; k < Nets && j+k < len(words); k++ {
				aw1, ab1, aw2, ab2 := aw1.Copy(), ab1.Copy(), aw2.Copy(), ab2.Copy()
				bw1, bb1, bw2, bb2 := bw1.Copy(), bb1.Copy(), bw2.Copy(), bb2.Copy()
				cp := []*tf32.V{
					&aw1, &ab1, &aw2, &ab2,
					&bw1, &bb1, &bw2, &bb2,
				}
				copies = append(copies, cp)
				go learn(cp, words[j+k])
				flight++
			}
			for j := 0; j < flight; j++ {
				total += <-done
			}

			for _, parameters := range copies {
				norm := float32(0)
				for _, p := range parameters {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
							p.X[l] += deltas[k][l]
						}
					}
				} else {
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d
							p.X[l] += deltas[k][l]
						}
					}
				}
			}
			fmt.Printf(".")
		}
		fmt.Printf("\n")

		set := Set{
			Cost:  float64(total),
			Epoch: uint64(i),
		}
		add := func(name string, w *tf32.V) {
			shape := make([]int64, len(w.S))
			for i := range shape {
				shape[i] = int64(w.S[i])
			}
			weights := Weights{
				Name:   name,
				Shape:  shape,
				Values: w.X,
			}
			set.Weights = append(set.Weights, &weights)
		}
		add("aw1", &aw1)
		add("ab1", &ab1)
		add("aw2", &aw2)
		add("ab2", &ab2)
		add("aw1", &bw1)
		add("ab1", &bb1)
		add("aw2", &bw2)
		add("ab2", &bb2)
		out, err := proto.Marshal(&set)
		if err != nil {
			panic(err)
		}
		output, err := os.Create(fmt.Sprintf("weights_%d.w", i))
		if err != nil {
			panic(err)
		}
		_, err = output.Write(out)
		if err != nil {
			panic(err)
		}
		output.Close()

		fmt.Println(i, total/float32(NumberOfVerses), time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

// VariableLearn learns the rnn model
func VariableLearn() {
	verses, _, _ := Verses()

	initial := tf32.NewV(2*Space, 1)
	initial.X = initial.X[:cap(initial.X)]
	w1, b1 := tf32.NewV(2*Width, Scale*2*Width), tf32.NewV(Scale*2*Width)
	w2, b2 := tf32.NewV(Scale*4*Width, Width), tf32.NewV(Width)
	parameters := []*tf32.V{&w1, &b1, &w2, &b2}
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, Random32(-1, 1))
		}
	}
	factor := float32(math.Sqrt(2 * Width))
	for i, x := range w1.X {
		w1.X[i] = x / factor
	}
	factor = float32(math.Sqrt(4 * Width))
	for i, x := range w2.X {
		w2.X[i] = x / factor
	}

	deltas := make([][]float32, 0, len(parameters))
	for _, p := range parameters {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	done := make(chan float32, 8)
	learn := func(parameters []*tf32.V, verse string) {
		w1, b1, w2, b2 := parameters[0], parameters[1], parameters[2], parameters[3]
		verseSymbols := []rune(verse)
		if len(verseSymbols) > 16 {
			verseSymbols = verseSymbols[:16]
		}
		symbols := make([]tf32.V, 0, len(verseSymbols))
		for _, s := range verseSymbols {
			symbol := tf32.NewV(2*Symbols, 1)
			symbol.X = symbol.X[:cap(symbol.X)]
			index := 2 & int(s)
			symbol.X[index] = 0
			symbol.X[index+1] = 1
			symbols = append(symbols, symbol)
		}

		l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[0].Meta(), initial.Meta())), b1.Meta()))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
		cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[1].Meta()))
		for j := 1; j < len(symbols)-1; j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[j].Meta(), tf32.Slice(l2, space.Meta()))), b1.Meta()))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
			cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[j+1].Meta())))
		}

		done <- tf32.Gradient(cost).X[0]
	}

	iterations := 100
	alpha, eta := float32(.3), float32(.3/float64(Nets))
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range verses {
			j := i + rand.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := float32(0.0)
		for j := 0; j < len(verses); j += Nets {
			flight, copies := 0, make([][]*tf32.V, 0, Nets)
			for k := 0; k < Nets && j+k < len(verses); k++ {
				w1, b1, w2, b2 := w1.Copy(), b1.Copy(), w2.Copy(), b2.Copy()
				cp := []*tf32.V{&w1, &b1, &w2, &b2}
				copies = append(copies, cp)
				go learn(cp, verses[j+k])
				flight++
			}
			for j := 0; j < flight; j++ {
				total += <-done
			}

			for _, parameters := range copies {
				norm := float32(0)
				for _, p := range parameters {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
							p.X[l] += deltas[k][l]
						}
					}
				} else {
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d
							p.X[l] += deltas[k][l]
						}
					}
				}
			}
			fmt.Printf(".")
		}
		fmt.Printf("\n")

		set := Set{
			Cost:  float64(total),
			Epoch: uint64(i),
		}
		add := func(name string, w *tf32.V) {
			shape := make([]int64, len(w.S))
			for i := range shape {
				shape[i] = int64(w.S[i])
			}
			weights := Weights{
				Name:   name,
				Shape:  shape,
				Values: w.X,
			}
			set.Weights = append(set.Weights, &weights)
		}
		add("w1", &w1)
		add("b1", &b1)
		add("w2", &w2)
		add("b2", &b2)
		out, err := proto.Marshal(&set)
		if err != nil {
			panic(err)
		}
		output, err := os.Create(fmt.Sprintf("weights_%d.w", i))
		if err != nil {
			panic(err)
		}
		_, err = output.Write(out)
		if err != nil {
			panic(err)
		}
		output.Close()

		fmt.Println(i, total/float32(NumberOfVerses), time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

// FixedLearn learns the rnn model
func FixedLearn() {
	verses, _, _ := Verses()
	max := Scale * 8

	symbols := make([][]tf32.V, Nets)
	for i := range symbols {
		symbols[i] = make([]tf32.V, 0, max)
		for j := 0; j < max; j++ {
			symbol := tf32.NewV(2*Symbols, Batch)
			symbol.X = symbol.X[:cap(symbol.X)]
			symbols[i] = append(symbols[i], symbol)
		}
	}
	initial := tf32.NewV(2*Space, Batch)
	for i := 0; i < cap(initial.X); i++ {
		initial.X = append(initial.X, 0)
	}
	w1, b1 := tf32.NewV(2*Width, Scale*2*Width), tf32.NewV(Scale*2*Width)
	w2, b2 := tf32.NewV(Scale*4*Width, Width), tf32.NewV(Width)
	parameters := []*tf32.V{&w1, &b1, &w2, &b2}
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, Random32(-1, 1))
		}
	}
	factor := float32(math.Sqrt(2 * Width))
	for i, x := range w1.X {
		w1.X[i] = x / factor
	}
	factor = float32(math.Sqrt(4 * Width))
	for i, x := range w2.X {
		w2.X[i] = x / factor
	}

	deltas := make([][][]float32, Nets)
	for i := range deltas {
		for _, p := range parameters {
			deltas[i] = append(deltas[i], make([]float32, len(p.X)))
		}
	}
	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[0][0].Meta(), initial.Meta())), b1.Meta()))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
	cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0][1].Meta()))
	for j := 1; j < max-1; j++ {
		l1 = tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[0][j].Meta(), tf32.Slice(l2, space.Meta()))), b1.Meta()))
		l2 = tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
		cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0][j+1].Meta())))
	}

	iterations := 100
	alpha, eta := float32(.3), float32(.3/float64(Nets))
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range verses {
			j := i + rand.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := float32(0)
		for i := 0; i < len(verses); i += Nets * Batch {
			fmt.Printf(".")
			for _, s := range symbols {
				for i := range s {
					s[i].Zero()
					for j := range s[i].X {
						if j%2 == 0 {
							s[i].X[j] = 0
						} else {
							s[i].X[j] = 0
						}
					}
				}
			}
			for j, symbols := range symbols {
				for k, verse := range verses[i+j*Batch : i+(j+1)*Batch] {
					if len(verse) > max {
						verse = verse[:max]
					}
					for l, symbol := range verse {
						index := 2 * (k*Symbols + int(symbol))
						symbols[l].X[index] = 0
						symbols[l].X[index+1] = 1
					}
				}
			}

			for _, p := range parameters {
				p.Zero()
			}

			costs := make([]tf32.Meta, Nets)
			params := [][]*tf32.V{parameters}
			for i := range costs {
				if i == 0 {
					costs[i] = cost
					continue
				}
				w1, b1 := w1.Copy(), b1.Copy()
				w2, b2 := w2.Copy(), b2.Copy()
				params = append(params, []*tf32.V{&w1, &b1, &w2, &b2})
				l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[i][0].Meta(), initial.Meta())), b1.Meta()))
				l2 := tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
				costs[i] = tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[i][1].Meta()))
				for j := 1; j < max-1; j++ {
					l1 = tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[i][j].Meta(), tf32.Slice(l2, space.Meta()))), b1.Meta()))
					l2 = tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
					costs[i] = tf32.Add(costs[i], tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[i][j+1].Meta())))
				}
			}

			done := make(chan float32, Nets)
			for _, cost := range costs {
				go func(cost tf32.Meta) {
					done <- tf32.Gradient(cost).X[0]
				}(cost)
			}
			for range costs {
				total += <-done / Batch
			}

			for i, parameters := range params {
				norm := float32(0)
				for _, p := range parameters {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[i][k][l] = alpha*deltas[i][k][l] - eta*d*scaling
							p.X[l] += deltas[i][k][l]
						}
					}
				} else {
					for k, p := range parameters {
						for l, d := range p.D {
							deltas[i][k][l] = alpha*deltas[i][k][l] - eta*d
							p.X[l] += deltas[i][k][l]
						}
					}
				}
			}
		}
		fmt.Printf("\n")

		set := Set{
			Cost:  float64(total),
			Epoch: uint64(i),
		}
		add := func(name string, w *tf32.V) {
			shape := make([]int64, len(w.S))
			for i := range shape {
				shape[i] = int64(w.S[i])
			}
			weights := Weights{
				Name:   name,
				Shape:  shape,
				Values: w.X,
			}
			set.Weights = append(set.Weights, &weights)
		}
		add("w1", &w1)
		add("b1", &b1)
		add("w2", &w2)
		add("b2", &b2)
		out, err := proto.Marshal(&set)
		if err != nil {
			panic(err)
		}
		output, err := os.Create(fmt.Sprintf("weights_%d.w", i))
		if err != nil {
			panic(err)
		}
		_, err = output.Write(out)
		if err != nil {
			panic(err)
		}
		output.Close()

		fmt.Println(i, total, time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}

// Verses gets the bible verses
func Verses() ([]string, []string, int) {
	testaments, verses, words, max :=
		Bible(), make([]string, 0, NumberOfVerses), make([]string, 0, 8), 0
	for _, testament := range testaments {
		if *FlagVerbose {
			fmt.Printf("%s\n\n", testament.Name)
		}
		for _, book := range testament.Books {
			if *FlagVerbose {
				fmt.Printf(" %s\n", book.Name)
			}
			for _, verse := range book.Verses {
				if *FlagVerbose {
					fmt.Printf("  %s %s\n", verse.Number, verse.Verse)
				}
				if length := len(verse.Verse); length > max {
					max = length
				}
				verses = append(verses, verse.Verse)
			}
			if *FlagVerbose {
				fmt.Printf("\n")
			}
		}
		if *FlagVerbose {
			fmt.Printf("\n")
		}
	}
	if len(verses) != NumberOfVerses {
		panic("wrong number of verses")
	}
	seen := make(map[string]bool)
	for _, verse := range verses {
		verseWords := PatternWord.Split(verse, -1)
		for _, word := range verseWords {
			word = strings.Trim(word, ".?!")
			if seen[word] {
				continue
			}
			seen[word] = true
			words = append(words, word)
		}
	}
	return verses, words, max
}

// Bible returns the bible
func Bible() []Testament {
	data, err := ioutil.ReadFile("pg10.txt")
	if err != nil {
		panic(err)
	}
	bible := string(data)
	beginning := strings.Index(bible, "*** START OF THIS PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE ***")
	ending := strings.Index(bible, "End of the Project Gutenberg EBook of The King James Bible")
	bible = bible[beginning:ending]
	testaments := make([]Testament, 2)
	testaments[0].Name = "The Old Testament of the King James Version of the Bible"
	testaments[1].Name = "The New Testament of the King James Bible"

	a := strings.Index(bible, testaments[0].Name)
	b := strings.Index(bible, testaments[1].Name)
	parse := func(t *Testament, testament string) {
		books := PatternBook.FindAllStringIndex(testament, -1)
		for i, book := range books {
			b := Book{
				Name: strings.TrimSpace(testament[book[0]:book[1]]),
			}
			end := len(testament)
			if i+1 < len(books) {
				end = books[i+1][0]
			}
			content := testament[book[1]:end]
			lines := PatternVerse.FindAllStringIndex(content, -1)
			for _, line := range lines {
				l := strings.TrimSpace(strings.ReplaceAll(content[line[0]:line[1]], "\r\n", " "))
				a := strings.Index(l, " ")
				verse := Verse{
					Number: strings.TrimSpace(l[:a]),
					Verse:  strings.TrimSpace(l[a:]),
				}
				b.Verses = append(b.Verses, verse)
			}
			t.Books = append(t.Books, b)
		}
	}
	parse(&testaments[0], bible[a:b])
	parse(&testaments[1], bible[b:])

	return testaments
}

// Random32 return a random float32
func Random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}
