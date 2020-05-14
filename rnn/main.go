// Copyright 2020 The Models Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strings"
)

var (
	// PatternBook marks the start of a book
	PatternBook = regexp.MustCompile(`\r\n\r\n\r\n\r\n[A-Za-z]+([ \t]+[A-Za-z:]+)*\r\n\r\n`)
	// PatternVerse is a verse
	PatternVerse = regexp.MustCompile(`\d+[:]\d+[A-Za-z:.,?;"' ()\t\r\n]+`)
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
	testaments, verses := Bible(), make([]string, 0, 8)
	for _, testament := range testaments {
		fmt.Printf("%s\n\n", testament.Name)
		for _, book := range testament.Books {
			fmt.Printf(" %s\n", book.Name)
			for _, verse := range book.Verses {
				fmt.Printf("  %s %s\n", verse.Number, verse.Verse)
				verses = append(verses, verse.Verse)
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}
	if len(verses) != 31102 {
		panic("wrong number of verses")
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
