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
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

const (
	// Symbols is the number of symbols
	Symbols = 256
	// Space is the state space of the Println
	Space = 256
	// Width is the width of the neural network
	Width = Symbols + Space
	// Batch is the batch size
	Batch = 256
)

var (
	// PatternBook marks the start of a book
	PatternBook = regexp.MustCompile(`\r\n\r\n\r\n\r\n[A-Za-z]+([ \t]+[A-Za-z:]+)*\r\n\r\n`)
	// PatternVerse is a verse
	PatternVerse = regexp.MustCompile(`\d+[:]\d+[A-Za-z:.,?;"' ()\t\r\n]+`)
	// FlagVerbose enables verbose mode
	FlagVerbose = flag.Bool("verbose", false, "verbose mode")
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

	testaments, verses, max := Bible(), make([]string, 0, 8), 0
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
	if len(verses) != 31102 {
		panic("wrong number of verses")
	}

	max = 8

	symbols := make([]tf32.V, 0, max)
	for i := 0; i < max; i++ {
		symbol := tf32.NewV(2*Symbols, Batch)
		for i := 0; i < cap(symbol.X); i++ {
			symbol.X = append(symbol.X, 0)
		}
		symbols = append(symbols, symbol)
	}
	initial := tf32.NewV(2*Space, Batch)
	for i := 0; i < cap(initial.X); i++ {
		initial.X = append(initial.X, 0)
	}
	w1, b1 := tf32.NewV(2*Width, 2*Width), tf32.NewV(2*Width)
	w2, b2 := tf32.NewV(4*Width, Width), tf32.NewV(Width)
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

	var deltas [][]float32
	for _, p := range parameters {
		deltas = append(deltas, make([]float32, len(p.X)))
	}
	symbol := tf32.NewV(2, 1)
	symbol.X = append(symbol.X, 0, 2*Symbols)
	space := tf32.NewV(2, 1)
	space.X = append(space.X, 2*Symbols, 2*Symbols+2*Space)

	l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[0].Meta(), initial.Meta())), b1.Meta()))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
	cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[1].Meta()))
	for i := 1; i < max-1; i++ {
		l1 = tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), tf32.Concat(symbols[i].Meta(), tf32.Slice(l2, space.Meta()))), b1.Meta()))
		l2 = tf32.Everett(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
		cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[i+1].Meta())))
	}

	iterations := 100
	alpha, eta := float32(.4), float32(.6)
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range verses {
			j := i + rand.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := float32(0)
		for i := 0; i < len(verses); i += Batch {
			fmt.Printf(".")
			for i := range symbols {
				symbols[i].Zero()
				for j := range symbols[i].X {
					if j%2 == 0 {
						symbols[i].X[j] = -1
					} else {
						symbols[i].X[j] = 0
					}
				}
			}
			for j, verse := range verses[i : i+Batch] {
				if len(verse) > max {
					verse = verse[:max]
				}
				for k, symbol := range verse {
					index := 2 * (j*Symbols + int(symbol))
					symbols[k].X[index] = 0
					symbols[k].X[index+1] = 1
				}
			}

			for _, p := range parameters {
				p.Zero()
			}

			total += tf32.Gradient(cost).X[0]

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
		fmt.Printf("\n")

		fmt.Println(total, time.Now().Sub(start))
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
