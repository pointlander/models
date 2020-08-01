package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/mnist"
	"github.com/pointlander/gradient/tf32"
)

const (
	// Width is the width of the neural network
	Width = mnist.Width * mnist.Height
)

// FlagTest test a model
var FlagTest = flag.String("test", "", "test a model")

func main() {
	rand.Seed(1)

	flag.Parse()

	datum, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	fmt.Println("images", len(datum.Train.Images))

	if *FlagTest != "" {
		set := tf32.NewSet()
		cost, epoch, err := set.Open(*FlagTest)
		if err != nil {
			panic(err)
		}
		fmt.Println(cost, epoch)

		correct := 0
		for i, testImage := range datum.Test.Images {
			image := tf32.NewV(Width, 1)
			image.X = image.X[:cap(image.X)]

			for j, value := range testImage {
				image.X[j] = float32(value) / 255
			}
			l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), image.Meta()), set.Get("ab1")))
			l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
			max, number := float32(0.0), 0
			l2(func(a *tf32.V) bool {
				for j, value := range a.X {
					if value > max {
						max, number = value, j
					}
				}
				return true
			})
			if (number-1)/2 == int(datum.Test.Labels[i]) {
				correct++
			}
		}
		fmt.Println(float64(correct) / float64(len(datum.Test.Images)))
		return
	}

	set := tf32.NewSet()
	set.Add("aw1", Width, 4*Width)
	set.Add("ab1", 4*Width)
	set.Add("aw2", 8*Width, 10)
	set.Add("ab2", 10)

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

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	image := tf32.NewV(Width, 100)
	image.X = image.X[:cap(image.X)]
	label := tf32.NewV(20, 100)
	label.X = label.X[:cap(label.X)]

	indexes := make([]int, len(datum.Train.Images))
	for i := range indexes {
		indexes[i] = i
	}

	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), image.Meta()), set.Get("ab1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
	cost := tf32.Avg(tf32.Quadratic(label.Meta(), l2))

	iterations := 30
	alpha, eta := float32(.3), float32(.3)
	points := make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for i := range indexes {
			j := i + rand.Intn(len(indexes)-i)
			indexes[i], indexes[j] = indexes[j], indexes[i]
		}

		total := float32(0.0)
		start := time.Now()
		for i := 0; i < len(indexes); i += 100 {
			weights := set.ByName["aw1"]
			weights.Seed = tf32.RNG(i + 1)
			weights.Drop = .5

			weights = set.ByName["ab1"]
			weights.Seed = tf32.RNG(i + 1)
			weights.Drop = .5

			set.Zero()
			image.Zero()
			label.Zero()
			index := 0
			for j := 0; j < 100; j++ {
				for _, value := range datum.Train.Images[indexes[i+j]] {
					image.X[index] = float32(value) / 255
					index++
				}
			}
			for j := range label.X {
				label.X[j] = 0
			}
			index = 0
			for j := 0; j < 100; j++ {
				label.X[index+2*int(datum.Train.Labels[indexes[i+j]])+1] = 1
				index += 20
			}

			total += tf32.Gradient(cost).X[0]
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
					if p.N == "aw1" {
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
					} else if p.N == "ab1" {
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

			if i%1000 == 0 {
				fmt.Print(".")
			}
		}
		fmt.Printf("\n")

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
		if err != nil {
			panic(err)
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total, time.Now().Sub(start))
		start = time.Now()
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
