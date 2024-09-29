package main

import (
	"fmt"
	"path/filepath"

	"github.com/kelindar/bert"
)

func main() {

	// https://huggingface.co/skeskinen/ggml/tree/main
	// https://arxiv.org/pdf/2210.17114
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

	// Batch encode
	texts := []string{
		"First sentence.",
		"Second sentence.",
	}
	batchEmbeddings, err := ctx.EmbedTextBatch(texts, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println("Batch Embeddings:", batchEmbeddings)

	// Tokenize
	tokens, err := ctx.Tokenize(text)
	if err != nil {
		panic(err)
	}
	fmt.Println("Tokens:", tokens)

	// Evaluate tokens
	embeddings, err = ctx.EmbedTokens(tokens)
	if err != nil {
		panic(err)
	}
	fmt.Println("Embeddings from tokens:", embeddings)

	// Evaluate tokens (batch)
	tokenEmbeddings, err := ctx.EmbedTokensBatch([][]bert.Token{tokens, tokens}, 2)
	if err != nil {
		panic(err)
	}

	fmt.Println("Batch embeddings from tokens:", tokenEmbeddings)
}
