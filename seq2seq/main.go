// Copyright 2020 The Models Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	"github.com/agnivade/levenshtein"
	"github.com/boltdb/bolt"
	"github.com/c-bata/go-prompt"
	"github.com/golang/protobuf/proto"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/bible"
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
	// FlagVerbose enables verbose mode
	FlagVerbose = flag.Bool("verbose", false, "verbose mode")
	// FlagLearn learn the model
	FlagLearn = flag.String("learn", "", "learning mode")
	// FlagGraph graphs the model files
	FlagGraph = flag.String("graph", "", "graph mode")
	// FlagWordsModel the words model
	FlagWordsModel = flag.String("wordsModel", "", "the words model file")
	// FlagPhrasesModel the phrases model
	FlagPhrasesModel = flag.String("phrasesModel", "", "the phrase seq2seq model")
	// FlagInference load weights and generate probable strings
	FlagInference = flag.String("inference", "", "inference mode")
	// FlagBuildDB build the vector db
	FlagBuildDB = flag.Bool("db", false, "build the vector db")
	// FlagWords test words seq2seq model
	FlagWords = flag.String("words", "", "test words seq2seq model")
	// FlagBrain test words with bfloat16
	FlagBrain = flag.Bool("brain", false, "test words with bfloat16")
	// FlagSearch search the bible
	FlagSearch = flag.Bool("search", false, "search the bible")
	// FlagSparse sparse mode
	FlagSparse = flag.Bool("sparse", false, "sparse mode")
	// FlagDropout dropout mode
	FlagDropout = flag.Bool("dropout", false, "dropout mode")
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
	Testament string
	Book      string
	Number    string
	Verse     string
}

func main() {
	flag.Parse()

	if *FlagLearn != "" {
		switch *FlagLearn {
		case "hierarchical":
			HierarchicalLearn(tf32.Everett)
		case "hierarchical_experiment":
			HierarchicalLearn(tf32.EverettReLu)
		case "hierarchical_sentence":
			HierarchicalSentenceLearn(*FlagWordsModel)
		}

		return
	} else if *FlagBuildDB {
		BuildVectorDB(*FlagWordsModel, *FlagPhrasesModel)
		return
	} else if *FlagSearch {
		Search(*FlagWordsModel, *FlagPhrasesModel)
		return
	} else if *FlagInference != "" {
		if *FlagWords == "everett" {
			WordsInference(tf32.Everett)
		} else if *FlagWords == "relu" {
			WordsInference(tf32.EverettReLu)
		}
		return
	} else if *FlagGraph != "" {
		Graph(*FlagGraph)
		return
	}

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	maxVerse, maxWords := 0, 0
	for _, verse := range verses {
		if length := len(verse.Verse); length > maxVerse {
			maxVerse = length
		}
		if length := len(verse.Words); length > maxWords {
			maxWords = length
		}
	}
	sentences := bible.GetSentences()
	words := bible.GetWords()
	maxWord := 0
	for _, word := range words {
		if length := len(word); length > maxWord {
			maxWord = length
		}
	}
	fmt.Printf("number of verses %d\n", len(verses))
	fmt.Printf("number of sentences %d\n", len(sentences))
	fmt.Printf("max verse length %d\n", maxVerse)
	fmt.Printf("max words in verse %d\n", maxWords)
	fmt.Printf("number of unique words %d\n", len(words))
	fmt.Printf("max word length %d\n", maxWord)
	return
}

// Graph graphs the weight files
func Graph(directory string) {
	input, err := os.Open(directory)
	if err != nil {
		panic(err)
	}
	defer input.Close()
	type Pair struct {
		Epoch int
		Cost  float32
	}
	pairs := make(map[int]*Pair)
	names, err := input.Readdirnames(-1)
	if err != nil {
		panic(err)
	}
	for _, name := range names {
		if strings.HasSuffix(name, ".w") {
			set := tf32.NewSet()
			cost, epoch, err := set.Open(path.Join(directory, name))
			if err != nil {
				panic(err)
			}
			pair := Pair{
				Epoch: epoch,
				Cost:  cost,
			}
			pairs[epoch] = &pair
		}
	}

	points := make(plotter.XYs, 0, len(pairs))
	for _, pair := range pairs {
		points = append(points, plotter.XY{X: float64(pair.Epoch), Y: float64(pair.Cost)})
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

// Dot32 computes the 32 bit dot product
func Dot32(X, Y []float32) float32 {
	var sum float32
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

// Sqrt32 computes the 32 bit square root
func Sqrt32(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

// Similarity computes the cosine similarity
func Similarity(a, b []float32) float32 {
	if a == nil || b == nil {
		return 0
	}
	if len(a) != len(b) {
		panic(fmt.Errorf("vectors are mismatched %d != %d", len(a), len(b)))
	}
	aa, bb, ab := Dot32(a, a), Dot32(b, b), Dot32(a, b)
	return ab / (Sqrt32(aa) * Sqrt32(bb))
}

// Search search the bible
func Search(wordsModel, phrasesModel string) {
	book, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := book.GetVerses()
	fmt.Println("verses", len(verses))

	options := bolt.Options{
		ReadOnly: true,
	}
	db, err := bolt.Open("vectors.db", 0600, &options)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	set1 := tf32.NewSet()
	cost, epoch, err := set1.Open(wordsModel)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	encode1 := func(word string) *tf32.V {
		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set1.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set1.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set1.Get("aw2"), l1), set1.Get("ab2")))
		for _, s := range word {
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

		return &state
	}

	set2 := tf32.NewSet()
	cost, epoch, err = set2.Open(phrasesModel)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	encode2 := func(words []tf32.V) *tf32.V {
		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set2.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set2.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set2.Get("aw2"), l1), set2.Get("ab2")))
		for _, word := range words {
			copy(symbol.X, word.X)
			l2(func(a *tf32.V) bool {
				copy(state.X, a.X)
				return true
			})
		}

		return &state
	}

	completer := func(document prompt.Document) []prompt.Suggest {
		suggest := []prompt.Suggest{
			{Text: "exit", Description: "Exit the system"},
			{Text: "syn", Description: "find synonyms"},
		}
		return prompt.FilterHasPrefix(suggest, document.GetWordBeforeCursor(), true)
	}

	type Result struct {
		Similarity float32
		Verse      uint64
		Word       string
	}

	for {
		query := prompt.Input("> ", completer)
		parts := strings.Split(query, " ")
		if len(parts) < 1 {
			continue
		}
		switch parts[0] {
		case "exit":
			return
		case "syn":
			encode := func(word string) []float32 {
				vector := encode1(word)
				v := make([]tf32.V, 1)
				v[0] = *vector
				wordVector := encode2(v)
				return wordVector.X
			}
			if len(parts) != 2 {
				fmt.Println("word required")
				break
			}
			encoded := encode(parts[1])

			var results [10]Result
			for i := range results {
				results[i].Similarity = -1
			}
			err = db.View(func(tx *bolt.Tx) error {
				bucket := tx.Bucket([]byte("words"))
				cursor := bucket.Cursor()
				key, value := cursor.First()
				for key != nil && value != nil {
					vector := Vector{}
					err := proto.Unmarshal(value, &vector)
					if err != nil {
						return err
					}
					values := make([]float32, len(vector.Vector))
					for i, value := range vector.Vector {
						values[i] = math.Float32frombits(value << 16)
					}
					similarity := Similarity(encoded, values)
					for j := range results {
						if similarity > results[j].Similarity {
							if j > 0 {
								results[j-1] = results[j]
							}
							results[j].Similarity = similarity
							results[j].Word = string(key)
						}
					}
					key, value = cursor.Next()
				}
				return nil
			})
			if err != nil {
				panic(err)
			}
			for _, result := range results {
				fmt.Println(result.Similarity)
				fmt.Println(result.Word)
			}
		default:
			words := bible.PatternWord.Split(query, -1)
			symbols := make([]tf32.V, 0, len(words))
			for _, word := range words {
				word = strings.Trim(word, bible.WordCutSet)
				if word == "" {
					continue
				}
				encoding := encode1(word)
				symbols = append(symbols, *encoding)
			}
			encoded := encode2(symbols)

			var results [10]Result
			for i := range results {
				results[i].Similarity = -1
			}
			err = db.View(func(tx *bolt.Tx) error {
				bucket := tx.Bucket([]byte("vectors"))
				cursor := bucket.Cursor()
				key, value := cursor.First()
				for key != nil && value != nil {
					vector := Vector{}
					err := proto.Unmarshal(value, &vector)
					if err != nil {
						return err
					}
					values := make([]float32, len(vector.Vector))
					for i, value := range vector.Vector {
						values[i] = math.Float32frombits(value << 16)
					}
					similarity := Similarity(encoded.X, values)
					for i := range results {
						if similarity > results[i].Similarity {
							if i > 0 {
								results[i-1] = results[i]
							}
							results[i].Similarity = similarity
							results[i].Verse = vector.Verse
						}
					}
					key, value = cursor.Next()
				}
				return nil
			})
			if err != nil {
				panic(err)
			}
			for _, result := range results {
				fmt.Println(result.Similarity)
				fmt.Println(verses[result.Verse].Testament)
				fmt.Println(verses[result.Verse].Book)
				fmt.Println(verses[result.Verse].Verse)
			}
		}
	}
}

// BuildVectorDB builds the vector database
func BuildVectorDB(wordsModel, phrasesModel string) {
	start := time.Now()
	book, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := book.GetVerses()
	words := book.GetWords()
	fmt.Println("verses", len(verses))
	fmt.Println("words", len(words))

	set1 := tf32.NewSet()
	cost, epoch, err := set1.Open(wordsModel)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	encode1 := func(word string) *tf32.V {
		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set1.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set1.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set1.Get("aw2"), l1), set1.Get("ab2")))
		for _, s := range word {
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

		return &state
	}

	encoded := make(map[string]*tf32.V, len(words))
	for _, word := range words {
		encoded[word] = encode1(word)
	}

	set2 := tf32.NewSet()
	cost, epoch, err = set2.Open(phrasesModel)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	encode2 := func(words []tf32.V) *tf32.V {
		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set2.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set2.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set2.Get("aw2"), l1), set2.Get("ab2")))
		for _, word := range words {
			copy(symbol.X, word.X)
			l2(func(a *tf32.V) bool {
				copy(state.X, a.X)
				return true
			})
		}

		return &state
	}

	db, err := bolt.Open("vectors.db", 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	err = db.Update(func(tx *bolt.Tx) error {
		bucket, err := tx.CreateBucket([]byte("words"))
		if err != nil {
			return err
		}
		for word, v := range encoded {
			vector := make([]tf32.V, 1)
			vector[0] = *v
			encoded := encode2(vector)
			values := make([]uint32, len(encoded.X))
			for j, value := range encoded.X {
				values[j] = math.Float32bits(value) >> 16
			}
			message := Vector{
				Verse:  0,
				Vector: values,
			}
			value, err := proto.Marshal(&message)
			if err != nil {
				return err
			}
			err = bucket.Put([]byte(word), value)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		panic(err)
	}

	err = db.Update(func(tx *bolt.Tx) error {
		bucket, err := tx.CreateBucket([]byte("vectors"))
		if err != nil {
			return err
		}
		for i, verse := range verses {
			phrases := bible.PatternSentence.Split(verse.Verse, -1)
			for _, phrase := range phrases {
				phrase = strings.Trim(phrase, bible.WordCutSet)
				if len(phrase) == 0 {
					continue
				}
				words := bible.PatternWord.Split(phrase, -1)
				symbols := make([]tf32.V, 0, len(words))
				for _, word := range words {
					word = strings.Trim(word, bible.WordCutSet)
					if word == "" {
						continue
					}
					state := tf32.NewV(2*Space, 1)
					state.X = state.X[:cap(state.X)]
					if encoding, ok := encoded[word]; ok {
						copy(state.X, encoding.X)
					} else {
						fmt.Println("not found", word)
					}
					symbols = append(symbols, state)
				}
				encoded := encode2(symbols)
				values := make([]uint32, len(encoded.X))
				for j, value := range encoded.X {
					values[j] = math.Float32bits(value) >> 16
				}
				vector := Vector{
					Verse:  uint64(i),
					Vector: values,
				}
				value, err := proto.Marshal(&vector)
				if err != nil {
					return err
				}
				sequence, err := bucket.NextSequence()
				if err != nil {
					return err
				}
				key := make([]byte, 8)
				binary.LittleEndian.PutUint64(key, sequence)
				err = bucket.Put(key, value)
				if err != nil {
					return err
				}
			}
			fmt.Println(i)
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("Real Time = %s\n", time.Now().Sub(start).String())
}

// WordsInference test words seq2seq
func WordsInference(activation func(a tf32.Meta) tf32.Meta) {
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	words := bible.GetWords()
	fmt.Println(len(words))

	set := tf32.NewSet()
	cost, epoch, err := set.Open(*FlagInference)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	sum, sumAbs, sumOfSquares, count := float32(0.0), float32(0.0), float32(0.0), 0
	min, max := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	small := 0
	histogram := make([]int, 0, 8)
	for _, weights := range set.Weights {
		last := 0
		for i, w := range weights.X {
			sum += w
			aw := w
			if aw < 0 {
				aw = -aw
			}
			sumAbs += aw
			sumOfSquares += w * w
			count++
			if w < min {
				min = w
			}
			if w > max {
				max = w
			}
			if float64(aw) < 1.0/(1<<15) {
				small++
				if *FlagSparse {
					weights.X[i] = 0
				}
				if i != 0 {
					diff := i - last
					if diff >= len(histogram) {
						n := make([]int, diff+1)
						copy(n, histogram)
						histogram = n
					}
					histogram[diff]++
				}
				last = i
			}
		}
	}
	average := sum / float32(count)
	fmt.Println("average", average)
	fmt.Println("abs average", sumAbs/float32(count))
	fmt.Println("variance", sumOfSquares/float32(count)-average*average)
	fmt.Println("min", min)
	fmt.Println("max", max)
	fmt.Println("small", float32(small)/float32(count))
	//fmt.Println("histogram", histogram)

	points := make(plotter.XYs, 0, 256)
	weights := set.ByName["aw1"]
	magnitudes := make([]float32, 2*weights.S[1])
	for i := 0; i < weights.S[1]; i++ {
		for j := 0; j < weights.S[0]; j++ {
			w := weights.X[i*weights.S[0]+j]
			magnitudes[i<<1] += w * w
			magnitudes[i<<1+1] += w * w
		}
	}
	for i, magnitude := range magnitudes {
		magnitudes[i] = float32(math.Sqrt(float64(magnitude)))
	}
	weights = set.ByName["aw2"]
	for i := 0; i < weights.S[1]; i++ {
		for j := 0; j < weights.S[0]; j++ {
			w := weights.X[i*weights.S[0]+j]
			points = append(points, plotter.XY{X: float64(magnitudes[j]), Y: float64(w)})
		}
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "weights vs weights"
	p.X.Label.Text = "weight"
	p.Y.Label.Text = "weight"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "weights.png")
	if err != nil {
		panic(err)
	}

	if *FlagBrain {
		for _, weights := range set.Weights {
			for i, w := range weights.X {
				bits := math.Float32bits(w)
				x := uint16(bits >> 16)
				weights.X[i] = math.Float32frombits(uint32(x) << 16)
			}
		}
	}

	autoencode := func(word string) string {
		autoencoded := ""

		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := activation(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), set.Get("ab1")))
		l2 := activation(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
		for _, s := range word {
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

		l1 = activation(tf32.Add(tf32.Mul(set.Get("bw1"), state.Meta()), set.Get("bb1")))
		l2 = activation(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
		for range word {
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
				autoencoded += fmt.Sprintf("%c", maxSymbol)
				return true
			})
		}
		return autoencoded
	}

	correct, distance := 0, 0
	for _, word := range words {
		autoencoded := autoencode(word)
		if autoencoded == word {
			correct++
		}
		distance += levenshtein.ComputeDistance(word, autoencoded)
	}
	fmt.Println(
		float64(correct)/float64(len(words)),
		float64(distance)/float64(len(words)-correct),
	)
}

// HierarchicalLearn learns the hierarchical encoder decoder rnn model for words
func HierarchicalLearn(activation func(a tf32.Meta) tf32.Meta) {
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	words := bible.GetWords()
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
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.NormFloat64())*factor)
			}
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

	type Completion struct {
		Cost float32
		Set  *tf32.Set
	}
	done, seed := make(chan Completion, 8), int64(0)
	learn := func(set *tf32.Set, seed int64, word string) {
		wordSymbols := []rune(word)
		symbols := make([]tf32.V, 0, len(wordSymbols))
		for _, s := range wordSymbols {
			symbol := tf32.NewV(2*Symbols, 1)
			symbol.X = symbol.X[:cap(symbol.X)]
			index := 2 * int(s)
			symbol.X[index] = 0
			symbol.X[index+1] = 1
			symbols = append(symbols, symbol)
		}

		if *FlagDropout {
			weights := set.ByName["aw1"]
			weights.Seed = tf32.RNG(seed)
			weights.Drop = .2

			weights = set.ByName["ab1"]
			weights.Seed = tf32.RNG(seed)
			weights.Drop = .2

			weights = set.ByName["bw1"]
			weights.Seed = tf32.RNG(seed)
			weights.Drop = .2

			weights = set.ByName["bb1"]
			weights.Seed = tf32.RNG(seed)
			weights.Drop = .2
		}

		l1 := activation(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbols[0].Meta(), initial.Meta())), set.Get("ab1")))
		l2 := activation(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
		for j := 1; j < len(symbols); j++ {
			l1 = activation(tf32.Add(tf32.Mul(set.Get("aw1"), tf32.Concat(symbols[j].Meta(), l2)), set.Get("ab1")))
			l2 = activation(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
		}

		l1 = activation(tf32.Add(tf32.Mul(set.Get("bw1"), l2), set.Get("bb1")))
		l2 = activation(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
		cost := tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[0].Meta()))
		for j := 1; j < len(symbols); j++ {
			l1 = activation(tf32.Add(tf32.Mul(set.Get("bw1"), tf32.Slice(l2, space.Meta())), set.Get("bb1")))
			l2 = activation(tf32.Add(tf32.Mul(set.Get("bw2"), l1), set.Get("bb2")))
			cost = tf32.Add(cost, tf32.Avg(tf32.Quadratic(tf32.Slice(l2, symbol.Meta()), symbols[j].Meta())))
		}

		if *FlagSparse {
			scale := tf32.NewV(1, 1)
			scale.X = append(scale.X, float32(Symbols*len(symbols))/
				float32(32*2*Width*Scale*2*Width+Scale*4*Width*Space+
					2*Space*Scale*2*Width+Scale*4*Width+Width))
			regularization := tf32.Add(tf32.Sum(tf32.Abs(set.Get("aw1"))), tf32.Sum(tf32.Abs(set.Get("aw2"))))
			regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("bw1"))))
			regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("bw2"))))
			regularization = tf32.Hadamard(scale.Meta(), regularization)
			cost = tf32.Add(cost, regularization)
		}

		done <- Completion{
			Cost: tf32.Gradient(cost).X[0],
			Set:  set,
		}
	}

	iterations := 200
	alpha, eta := float32(.3), float32(.3/float64(Nets))
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	update := func(set *tf32.Set) {
		norm := float32(0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for k, p := range set.Weights {
			if p.Seed != 0 {
				if p.N == "aw1" || p.N == "bw1" {
					rng, dropout := p.Seed, uint32((1-p.Drop)*math.MaxUint32)
					for l := 0; l < len(p.D); l += p.S[0] {
						if rng.Next() > dropout {
							continue
						}
						for m, d := range p.D[l : l+p.S[0]] {
							deltas[k][l+m] = alpha*deltas[k][l+m] - eta*d*scaling
							p.X[l+m] += deltas[k][l+m]
						}
					}
				} else if p.N == "ab1" || p.N == "bb1" {
					index, dropout := 0, uint32((1-p.Drop)*math.MaxUint32)
					for i := 0; i < p.S[1]; i++ {
						rng := p.Seed
						for j := 0; j < p.S[0]; j++ {
							if rng.Next() > dropout {
								index++
								continue
							}
							deltas[k][index] = alpha*deltas[k][index] - eta*p.D[index]*scaling
							p.X[index] += deltas[k][index]
							index++
						}
					}
				}
			} else {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
					p.X[l] += deltas[k][l]
				}
			}
		}
	}
	for i := 0; i < iterations; i++ {
		for i := range words {
			j := i + rand.Intn(len(words)-i)
			words[i], words[j] = words[j], words[i]
		}

		total := float32(0.0)
		j, flight := 0, 0
		for j < Nets && j < len(words) {
			word := words[j]
			cp := set.Copy()
			seed++
			go learn(&cp, seed, word)
			flight++
			j++
		}
		for j < len(words) {
			completion := <-done
			flight--
			total += completion.Cost
			update(completion.Set)
			word := words[j]
			cp := set.Copy()
			seed++
			go learn(&cp, seed, word)
			flight++
			j++
			if j%Nets == 0 {
				fmt.Printf(".")
			}
		}
		for j := 0; j < flight; j++ {
			completion := <-done
			total += completion.Cost
			update(completion.Set)
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

// HierarchicalSentenceLearn learns the hierarchical encoder decoder rnn model
// for sentences
func HierarchicalSentenceLearn(wordsModel string) {
	book, err := bible.Load()
	if err != nil {
		panic(err)
	}
	sentences := book.GetSentences()
	words := book.GetWords()
	fmt.Println("words", len(words))
	fmt.Println("sentences", len(sentences))

	setL1 := tf32.NewSet()
	cost, epoch, err := setL1.Open(wordsModel)
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epoch)

	encode := func(word string) *tf32.V {
		symbol := tf32.NewV(2*Symbols, 1)
		symbol.X = symbol.X[:cap(symbol.X)]
		state := tf32.NewV(2*Space, 1)
		state.X = state.X[:cap(state.X)]

		l1 := tf32.Everett(tf32.Add(tf32.Mul(setL1.Get("aw1"), tf32.Concat(symbol.Meta(), state.Meta())), setL1.Get("ab1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(setL1.Get("aw2"), l1), setL1.Get("ab2")))
		for _, s := range word {
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

		return &state
	}

	encoded := make(map[string]*tf32.V, len(words))
	for _, word := range words {
		encoded[word] = encode(word)
	}

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
		w := set.Weights[i]
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
	learn := func(set *tf32.Set, sentence string) {
		words := bible.PatternWord.Split(sentence, -1)
		symbols := make([]tf32.V, 0, len(words))
		for _, word := range words {
			word = strings.Trim(word, bible.WordCutSet)
			if word == "" {
				continue
			}
			state := tf32.NewV(2*Space, 1)
			state.X = state.X[:cap(state.X)]
			if encoding, ok := encoded[word]; ok {
				copy(state.X, encoding.X)
			} else {
				fmt.Println("not found", word)
			}
			symbols = append(symbols, state)
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

	iterations := 200
	fmt.Println("learning...")
	alpha, eta := float32(.3), float32(.3/float64(Nets))
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for i := range sentences {
			j := i + rand.Intn(len(sentences)-i)
			sentences[i], sentences[j] = sentences[j], sentences[i]
		}

		total := float32(0.0)
		for j := 0; j < len(sentences); j += Nets {
			flight, copies := 0, make([]*tf32.Set, 0, Nets)
			for k := 0; k < Nets && j+k < len(sentences); k++ {
				sentence := sentences[j+k]
				cp := set.Copy()
				copies = append(copies, &cp)
				go learn(&cp, sentence)
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

		err := set.Save(fmt.Sprintf("weights_sentence_%d.w", i), total, i)
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs_sentence.png")
	if err != nil {
		panic(err)
	}
}

// Random32 return a random float32
func Random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}
