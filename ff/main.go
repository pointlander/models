package main

import (
	"fmt"
	"math"
	"math/rand"

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

func main() {
	rand.Seed(1)

	datum, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	fmt.Println("images", len(datum.Train.Images))

	set := tf32.NewSet()
	set.Add("aw1", Width, 2*Width)
	set.Add("ab1", 2*Width)
	set.Add("aw2", 4*Width, 10)
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

	image := tf32.NewV(Width, 1)
	image.X = image.X[:cap(image.X)]
	label := tf32.NewV(20, 1)
	label.X = label.X[:cap(label.X)]

	indexes := make([]int, len(datum.Train.Images))
	for i := range indexes {
		indexes[i] = i
	}

	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw1"), image.Meta()), set.Get("ab1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("aw2"), l1), set.Get("ab2")))
	cost := tf32.Quadratic(label.Meta(), l2)

	iterations := 10
	alpha, eta := float32(.3), float32(.3)
	points := make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for i := range indexes {
			j := i + rand.Intn(len(indexes)-i)
			indexes[i], indexes[j] = indexes[j], indexes[i]
		}

		total := float32(0.0)
		for i, index := range indexes {
			weights := set.ByName["aw1"]
			weights.Seed = tf32.RNG(i + 1)
			weights.Drop = .5

			weights = set.ByName["ab1"]
			weights.Seed = tf32.RNG(i + 1)
			weights.Drop = .5

			set.Zero()
			image.Zero()
			label.Zero()
			for j, value := range datum.Train.Images[index] {
				image.X[j] = float32(value) / 255
			}
			for j := range label.X {
				label.X[j] = 0
			}
			label.X[2*datum.Train.Labels[index]+1] = 1

			t := tf32.Gradient(cost).X[0]
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

			total += t
			if i%1000 == 0 {
				fmt.Println(i, total)
			}
		}

		err := set.Save(fmt.Sprintf("weights_%d.w", i), total, i)
		if err != nil {
			panic(err)
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(total)

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
