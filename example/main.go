package main

import (
	"fmt"
	"path/filepath"

	"github.com/kelindar/bert"
)

func main() {
	// https://huggingface.co/skeskinen/ggml/tree/main
	model, _ := filepath.Abs("../dist/minilm12-q4.bin")

	ctx, err := bert.New(model)
	if err != nil {
		panic(err)
	}
	defer ctx.Close()

	text := "This is a test sentence."

	// Single encode
	embeddings, err := ctx.EmbedText(text)
	if err != nil {
		panic(err)
	}
	fmt.Println("Embeddings:", embeddings)

}
