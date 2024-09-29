package bert

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

// BenchmarkBert/encode-24         	     314	   3664379 ns/op	    1648 B/op	       3 allocs/op
func BenchmarkBert(b *testing.B) {
	ctx := loadModel()
	defer ctx.Close()

	text := "This is a test sentence we are going to generate embeddings for."

	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ctx.EmbedText(text)
			assert.NoError(b, err)
		}
	})

	b.Run("batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ctx.EmbedTextBatch([]string{
				text, text, text, text, text,
				text, text, text, text, text,
			}, 10)
			assert.NoError(b, err)
		}
	})

}

func loadModel() *Bert {
	mod, _ := filepath.Abs("dist/minilm12-q4.bin")
	ctx, err := New(mod)
	if err != nil {
		panic(err)
	}
	return ctx
}

func TestEmbedTokens(t *testing.T) {
	ctx := loadModel()
	defer ctx.Close()

	tests := []string{
		"I like watermelons!",
		"I also enjoy pineapples.",
		"Je préfère les pommes.",
		"Ich esse gerne Äpfel.",
		"I like apples too.",
	}

	for _, text := range tests {
		tokens, err := ctx.Tokenize(text)
		assert.NoError(t, err)
		assert.NotEmpty(t, tokens)

		v1, err := ctx.EmbedText("This is a test sentence.")
		assert.NoError(t, err)
		assert.Equal(t, 384, len(v1))

		v2, err := ctx.EmbedTokens(tokens)
		assert.NoError(t, err)
		assert.Equal(t, 384, len(v2))

		// Compare the embeddings
		assert.LessOrEqual(t, cosine(v1, v2), float32(.25),
			text,
		)
	}
}

func TestEmbedBatch(t *testing.T) {
	ctx := loadModel()
	defer ctx.Close()

	input := []string{
		"Je préfère les pommes.",
		"Ich esse gerne Äpfel.",
		"私はりんごが好きです。",
		"I like apples too.",
	}

	out, err := ctx.EmbedTextBatch(input, 8)
	assert.NoError(t, err)

	// make sure that pairwise cosine distance is low
	for i := 0; i < len(out); i++ {
		single, err := ctx.EmbedText(input[i])
		assert.NoError(t, err)
		assert.Equal(t, len(single), len(out[i]))

		for j := i + 1; j < len(out); j++ {
			assert.LessOrEqual(t, cosine(out[i], out[j]), float32(.25),
				fmt.Sprintf(`"%s" vs "%s"`, input[i], input[j]),
			)
		}
	}
}

func TestToken(t *testing.T) {
	ctx := loadModel()
	defer ctx.Close()

	assert.Equal(t, 512, ctx.MaxTokens())
	assert.Equal(t, 384, ctx.Size())

	text := "Hello"
	tokens, err := ctx.Tokenize(text)
	assert.NoError(t, err)
	assert.NotEmpty(t, tokens)
	assert.Equal(t, 3, len(tokens))

	{
		s1, err := ctx.TokenString(tokens[0])
		assert.NoError(t, err)
		assert.Equal(t, "[CLS]", s1)
	}

	{
		s2, err := ctx.TokenString(tokens[1])
		assert.NoError(t, err)
		assert.Equal(t, "hello", s2)
	}

	{
		s3, err := ctx.TokenString(tokens[2])
		assert.NoError(t, err)
		assert.Equal(t, "[SEP]", s3)
	}
}

func TestFindLibrary_Windows(t *testing.T) {
	path, err := findLibrary("bert.dll", "windows", "dist/win-x64")
	assert.NotEmpty(t, path)
	assert.NoError(t, err)
}

func TestFindLibrary_Linux(t *testing.T) {
	path, err := findLibrary("libbert.so", "linux", "dist/linux-x64")
	assert.NotEmpty(t, path)
	assert.NoError(t, err)
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return dot / (na * nb)
}
