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
	"regexp"
	"runtime"
	"strings"
	"time"

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
	// FlagWords test words seq2seq model
	FlagWords = flag.String("words", "", "test words seq2seq model")
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
		if *FlagWords != "" {
			WordsInference()
		} else {
			Inference()
		}
		return
	} else {
		verses, words, max, maxWords := Verses()
		maxWord := 0
		for _, word := range words {
			if length := len(word); length > maxWord {
				maxWord = length
			}
		}
		fmt.Printf("number of verses %d\n", len(verses))
		fmt.Printf("max verse length %d\n", max)
		fmt.Printf("max words in verse %d\n", maxWords)
		fmt.Printf("number of unique words %d\n", len(words))
		fmt.Printf("max word length %d\n", maxWord)
		return
	}
}

// WordsInference test words seq2seq
func WordsInference() {
	set := tf32.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)
	symbol := tf32.NewV(2*Symbols, 1)
	symbol.X = symbol.X[:cap(symbol.X)]
	state := tf32.NewV(2*Space, 1)
	state.X = state.X[:cap(state.X)]

	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set.Get("ab1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
	for _, s := range *FlagWords {
		for i := range symbol.X {
			symbol.X[i] = 0
		}
		index := 2 * int(s)
		symbol.X[index] = 0
		symbol.X[index+1] = 1
		l2(func(a *tf32.V) bool {
			copy(state.X, a.X)
			return true
		})
	}

	fmt.Printf("'")
	l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw1"), state.Meta()), set.Get("bb1")))
	l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
	for range *FlagWords {
		l2(func(a *tf32.V) bool {
			symbols := a.X[:2*Symbols]
			copy(state.X, a.X[2*Symbols:])
			max, maxSymbol := float32(0), rune('a')
			for i, symbol := range symbols {
				if i&1 == 1 {
					if symbol > max {
						max, maxSymbol = symbol, rune(i>>1)
					}
				}
			}
			fmt.Printf("%c", maxSymbol)
			return true
		})
	}
	fmt.Printf("'\n")
}

// Inference inference mode
func Inference() {
	set := tf32.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)
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
		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(input.Meta(), previous.Meta())), set.Get("b1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
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
	_, words, _, _ := Verses()
	fmt.Println(len(words))
	initial := tf32.NewV(2*Space, 1)
	initial.X = initial.X[:cap(initial.X)]
	set := tf32.NewSet()
	set.Add("aw1", 2*Width, Scale*2*Width)
	set.Add("ab1", Scale*2*Width)
	set.Add("aw2", Scale*4*Width, Space)
	set.Add("ab2", Space)
	set.Add("bw1", 2*Space, Scale*2*Width)
	set.Add("bb1", Scale*2*Width)
	set.Add("bw2", Scale*4*Width, Width)
	set.Add("bb2", Width)
	for i := range set.Weights {
		w := &set.Weights[i]
		factor := float32(math.Sqrt(float64(w.S[0])))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, Random32(-1, 1)/factor)
		}
	}

	deltas := make([][]float32, 0, len(set.Weights))
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	done := make(chan float32, 8)
	learn := func(set *tf32.Set, word string) {
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

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbols[0].Meta(), initial.Meta())), set.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
		for j := 1; j < len(symbols); j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbols[j].Meta(), l2)), set.Get("ab1")))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
		}

		l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw1"), l2), set.Get("bb1")))
		l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
		cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0].Meta()))
		for j := 1; j < len(symbols); j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw1"), tf32.Slice(l2, space.Meta())), set.Get("bb1")))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
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
			flight, copies := 0, make([]*tf32.Set, 0, Nets)
			for k := 0; k < Nets && j+k < len(words); k++ {
				word := words[j+k]
				cp := set.Copy()
				copies = append(copies, &cp)
				go learn(&cp, word)
				flight++
			}
			for j := 0; j < flight; j++ {
				total += <-done
			}

			for _, set := range copies {
				norm := float32(0)
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range set.Weights {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
							p.X[l] += deltas[k][l]
						}
					}
				} else {
					for k, p := range set.Weights {
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

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
		if err != nil {
			panic(err)
		}

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
	verses, _, _, _ := Verses()

	initial := tf32.NewV(2*Space, 1)
	initial.X = initial.X[:cap(initial.X)]
	set := tf32.NewSet()
	set.Add("w1", 2*Width, Scale*2*Width)
	set.Add("b1", Scale*2*Width)
	set.Add("w2", Scale*4*Width, Width)
	set.Add("b2", Width)
	for i := range set.Weights {
		w := &set.Weights[i]
		factor := float32(math.Sqrt(float64(w.S[0])))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, Random32(-1, 1)/factor)
		}
	}

	deltas := make([][]float32, 0, len(set.Weights))
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	done := make(chan float32, 8)
	learn := func(set *tf32.Set, verse string) {
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

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[0].Meta(), initial.Meta())), set.Get("b1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
		cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[1].Meta()))
		for j := 1; j < len(symbols)-1; j++ {
			l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[j].Meta(), tf32.Slice(l2, space.Meta()))), set.Get("b1")))
			l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
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
			flight, copies := 0, make([]*tf32.Set, 0, Nets)
			for k := 0; k < Nets && j+k < len(verses); k++ {
				cp := set.Copy()
				copies = append(copies, &cp)
				go learn(&cp, verses[j+k])
				flight++
			}
			for j := 0; j < flight; j++ {
				total += <-done
			}

			for _, set := range copies {
				norm := float32(0)
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range set.Weights {
						for l, d := range p.D {
							deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
							p.X[l] += deltas[k][l]
						}
					}
				} else {
					for k, p := range set.Weights {
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

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
		if err != nil {
			panic(err)
		}

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
	verses, _, _, _ := Verses()
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
	set := tf32.NewSet()
	set.Add("w1", 2*Width, Scale*2*Width)
	set.Add("b1", Scale*2*Width)
	set.Add("w2", Scale*4*Width, Width)
	set.Add("b2", Width)
	for i := range set.Weights {
		w := &set.Weights[i]
		factor := float32(math.Sqrt(float64(w.S[0])))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, Random32(-1, 1)/factor)
		}
	}

	deltas := make([][][]float32, Nets)
	for i := range deltas {
		for _, p := range set.Weights {
			deltas[i] = append(deltas[i], make([]float32, len(p.X)))
		}
	}
	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[0][0].Meta(), initial.Meta())), set.Get("b1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
	cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0][1].Meta()))
	for j := 1; j < max-1; j++ {
		l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[0][j].Meta(), tf32.Slice(l2, space.Meta()))), set.Get("b1")))
		l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
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

			set.Zero()

			costs := make([]tf32.Meta, Nets)
			sets := []*tf32.Set{&set}
			for i := range costs {
				if i == 0 {
					costs[i] = cost
					continue
				}
				set := set.Copy()
				sets = append(sets, &set)
				l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[i][0].Meta(), initial.Meta())), set.Get("b1")))
				l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
				costs[i] = tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[i][1].Meta()))
				for j := 1; j < max-1; j++ {
					l1 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), tf32.Concat(symbols[i][j].Meta(), tf32.Slice(l2, space.Meta()))), set.Get("b1")))
					l2 = tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
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

			for _, set := range sets {
				norm := float32(0)
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 1 {
					scaling := 1 / norm
					for k, p := range set.Weights {
						for l, d := range p.D {
							deltas[i][k][l] = alpha*deltas[i][k][l] - eta*d*scaling
							p.X[l] += deltas[i][k][l]
						}
					}
				} else {
					for k, p := range set.Weights {
						for l, d := range p.D {
							deltas[i][k][l] = alpha*deltas[i][k][l] - eta*d
							p.X[l] += deltas[i][k][l]
						}
					}
				}
			}
		}
		fmt.Printf("\n")

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
		if err != nil {
			panic(err)
		}

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
func Verses() ([]string, []string, int, int) {
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
	seen, maxWords := make(map[string]bool), 0
	for _, verse := range verses {
		verseWords := PatternWord.Split(verse, -1)
		if length := len(verseWords); length > maxWords {
			maxWords = length
		}
		for _, word := range verseWords {
			word = strings.Trim(word, ".?!")
			if seen[word] || len(word) == 0 {
				continue
			}
			seen[word] = true
			words = append(words, word)
		}
	}
	return verses, words, max, maxWords
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
